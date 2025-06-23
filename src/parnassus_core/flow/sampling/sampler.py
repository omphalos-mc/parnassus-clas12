from typing import Optional

import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm


class Sampler(ABC):
    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        reverse_time=False,
    ):
        """
        Args:
            n_steps: Number of steps to take
            use_edm_schedule: Whether to use the EDM schedule
            heavy_tail: Whether to use a heavy-tailed distribution
            zero_init_padded: Whether to zero out the padded values
            edm_config: Configuration for the EDM schedule (sigma_max, sigma_min, rho), required if using EDM schedule
            random_seed: Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.use_edm_schedule = use_edm_schedule
        self.heavy_tail = heavy_tail
        self.zero_init_padded = zero_init_padded
        self.reverse_time = reverse_time

        assert not use_edm_schedule or edm_config is not None, (
            "EDM config must be provided if using EDM schedule"
        )

        if self.use_edm_schedule:
            print("Using EDM schedule")
            idxs = torch.arange(n_steps)
            sigma_max = edm_config["sigma_max"]
            sigma_min = edm_config["sigma_min"]
            rho = edm_config["rho"]
            t_steps = (
                sigma_max ** (1 / rho)
                + idxs
                / (n_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            t_steps = torch.flip(t_steps, [0])
        else:
            t_steps = torch.linspace(0, 1, n_steps)
        self.t_steps = torch.cat([t_steps, torch.ones_like(t_steps)[:1]])
        if reverse_time:
            self.t_steps = torch.flip(self.t_steps, [0])

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        self.random_seed = random_seed

        if self.heavy_tail:
            self.distribution = torch.distributions.StudentT(7)
        else:
            self.distribution = torch.distributions.Normal(0, 1)

    def reset(self, random_seed=None):
        if random_seed is None:
            assert self.random_seed is not None, "Random seed not provided"
            random_seed = self.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _init_fastsim(self, shape, mask=None, device=torch.cpu):
        fastsim = self.distribution.sample(shape).to(device)
        if mask is not None and self.zero_init_padded and mask.shape[-1] == 2:
            fastsim[~mask[..., 1]] = 0
        return fastsim

    @torch.no_grad()
    def _transfer(self, x_curr, d_curr, dt):
        """Do the integration step.

        Args:
            x_curr: Current state
            d_curr: Current derivative
            dt: Time step

        Returns:
            x_next: Next state
        """
        return x_curr + d_curr * dt

    @torch.no_grad()
    def _step(
        self,
        model,
        fastsim,
        timestep,
        ctxt_dict,
        dt,
        **kwargs,
    ):
        """Do a single (euler) step of the model.

        Returns:
            fastsim_next: The next state of the fastsim
            deriv: Model output at the current timestep (dx/dt)

        """
        deriv = model.forward(
            fastsim,
            timestep=timestep.expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt)
        return fastsim_next, deriv

    @torch.no_grad()
    def _midpoint_step(
        self,
        model,
        fastsim,
        timestep,
        ctxt_dict,
        dt,
        **kwargs,
    ):
        """Do a single (midpoint) step of the model.

        Returns:
            fastsim_next: The next state of the fastsim
            deriv: Model output at the current timestep (dx/dt)

        """
        deriv = model.forward(
            fastsim,
            timestep=timestep.expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt / 2)
        deriv_next = model.forward(
            fastsim_next,
            timestep=(timestep + dt / 2).expand(fastsim.shape[0]),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv_next, dt)
        return fastsim_next, deriv_next

    @torch.no_grad()
    def _heun_step(
        self,
        model,
        fastsim,
        ctxt_dict,
        t_cur,
        t_next,
        is_last=False,
        **kwargs,
    ):
        fastsim_next, deriv_cur = self._step(
            model,
            fastsim,
            timestep=t_cur,
            dt=t_next - t_cur,
            ctxt_dict=ctxt_dict,
        )
        deriv_prev = deriv_cur
        if not is_last:
            _, deriv_prime = self._step(
                model,
                fastsim_next,
                timestep=t_next,
                dt=t_next - t_cur,
                ctxt_dict=ctxt_dict,
            )
            deriv_prev = (1 / 2) * (deriv_cur + deriv_prime)
            fastsim = self._transfer(fastsim, deriv_prev, t_next - t_cur)
        else:
            fastsim = fastsim_next

        return fastsim, deriv_prev

    @torch.no_grad()
    def _rk4_step(
        self,
        model,
        fastsim,
        t_list,
        *,
        ctxt_dict,
        **kwargs,
    ):
        x_2, e_1 = self._step(
            model,
            fastsim,
            timestep=t_list[0],
            dt=t_list[1] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        x_3, e_2 = self._step(
            model,
            x_2,
            timestep=t_list[1],
            dt=t_list[1] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        x_4, e_3 = self._step(
            model,
            x_3,
            timestep=t_list[1],
            dt=t_list[2] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        _, e_4 = self._step(
            model,
            x_4,
            timestep=t_list[2],
            dt=t_list[2] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        fastsim_next = self._transfer(fastsim, et, t_list[2] - t_list[0])
        return fastsim_next, et

    @abstractmethod
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        **kwargs,
    ):
        pass


class EulerSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        dtype, device = ctxt_dict["ctxt_data"].dtype, ctxt_dict["ctxt_data"].device
        if init_fastsim is None:
            fastsim = self._init_fastsim(pflow_shape, mask, device)
        else:
            fastsim = init_fastsim
        fastsim = fastsim.to(dtype)
        t_steps = self.t_steps.to(device).to(dtype)
        seq = [fastsim.cpu()]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._step(
                model,
                fastsim,
                timestep=t_cur,
                dt=t_next - t_cur,
                ctxt_dict=ctxt_dict,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            seq = torch.stack(seq)
            return fastsim.cpu(), seq
        return fastsim.cpu()


class MidpointSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        device = mask.device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        seq = [fastsim.cpu()]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._midpoint_step(
                model,
                fastsim,
                timestep=t_cur,
                dt=t_next - t_cur,
                ctxt_dict=ctxt_dict,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            seq = torch.stack(seq)
            return fastsim.cpu(), seq
        return fastsim.cpu()


class HeunSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        device = mask.device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._heun_step(
                model,
                fastsim,
                t_cur=t_cur,
                t_next=t_next,
                is_last=(i == self.n_steps - 1),
                ctxt_dict=ctxt_dict,
            )
        return fastsim.cpu()


class RK4Sampler(Sampler):
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        device = mask.device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._rk4_step(
                model,
                fastsim,
                mask,
                t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                ctxt_dict=ctxt_dict,
            )
        return fastsim.cpu()


class PNDMSampler(Sampler):
    def __init__(
        self,
        n_steps,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        init_step="rk4",
    ):
        """
        Args:
            n_steps: Number of steps to take
            use_edm_schedule: Whether to use the EDM schedule
            heavy_tail: Whether to use a heavy-tailed distribution
            zero_init_padded: Whether to zero out the padded values
            edm_config: Configuration for the EDM schedule (sigma_max, sigma_min, rho), required if using EDM schedule
            random_seed: Random seed for reproducibility
            init_step: Initial step to use (rk4, heun, euler)
        """
        self.init_step = init_step
        assert self.init_step in ["rk4", "heun", "euler"], "Invalid init_step"
        super().__init__(
            n_steps=n_steps,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
        )

        self.step_fn = {
            "rk4": self._rk4_step,
            "heun": self._heun_step,
            "euler": self._step,
        }[self.init_step]

    @torch.no_grad()
    def sample(
        self,
        model,
        pflow_shape,
        *,
        ctxt_dict,
        save_seq=False,
        **kwargs,
    ):
        mask = ctxt_dict["fs_mask"]
        device = mask.device
        fastsim = self._init_fastsim(pflow_shape, mask, device)
        t_steps = self.t_steps.to(device)
        ets = []
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            if len(ets) > 2:
                deriv_ = model.forward(
                    fastsim,
                    timestep=t_cur.expand(fastsim.shape[0]),
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_)
                deriv = (1 / 24) * (
                    55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]
                )
                fastsim = self._transfer(fastsim, deriv, t_next - t_cur)
                deriv_prev = deriv
            else:
                fastsim, deriv_prev = self.step_fn(
                    model,
                    fastsim,
                    t_cur=t_cur,
                    t_next=t_next,
                    t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                    is_last=(i == self.n_steps - 1),
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_prev)
            if len(ets) > 4:
                ets.pop(0)
        return fastsim.cpu()
