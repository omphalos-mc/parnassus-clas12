from typing import Optional

import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from parnassus_core.flow.utils.prediction import FlowPrediction


class Sampler(ABC):
    def __init__(
        self,
        model_fn,
        n_steps,
        prediction: FlowPrediction,
        use_edm_schedule: bool = False,
        heavy_tail: bool = False,
        zero_init_padded: bool = True,
        edm_config: dict | None = None,
        random_seed: int | None = None,
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
        self.model_fn = model_fn
        self.prediction = prediction
        self.n_steps = n_steps
        self.edm_config = edm_config
        self.use_edm_schedule = use_edm_schedule
        self.heavy_tail = heavy_tail
        self.zero_init_padded = zero_init_padded
        self.reverse_time = prediction.reverse_time

        # Init time
        self.init_time_steps(n_steps)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        self.random_seed = random_seed

        if self.heavy_tail:
            self.distribution = torch.distributions.StudentT(7)
        else:
            self.distribution = torch.distributions.Normal(0, 1)

    def init_time_steps(self, n_steps: int):
        if self.use_edm_schedule:
            print("Using EDM schedule")
            assert self.edm_config is not None, (
                "EDM config must be provided if using EDM schedule"
            )
            idxs = torch.arange(n_steps + 1)
            sigma_max = self.edm_config["sigma_max"]
            sigma_min = self.edm_config["sigma_min"]
            rho = self.edm_config["rho"]
            t_steps = (
                sigma_max ** (1 / rho)
                + idxs / n_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho
            t_steps = torch.flip(t_steps, [0])
            if self.reverse_time:
                t_steps = torch.flip(t_steps, [0])
        else:
            t_start, t_end = self.prediction.sampling_time_range()
            t_steps = torch.linspace(t_start, t_end, n_steps + 1)
        self.n_steps = n_steps
        self.t_steps = t_steps

    def reset(self, random_seed: int | None = None):
        if random_seed is None:
            assert self.random_seed is not None, "Random seed not provided"
            random_seed = self.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _init_fastsim(
        self,
        shape: tuple[int, ...] | torch.Size,
        mask: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        fastsim = self.distribution.sample(shape).to(device)
        if mask is not None and self.zero_init_padded and mask.shape[-1] == 2:
            fastsim[~mask[..., 1]] = 0
        return fastsim

    @torch.no_grad()
    def _transfer(
        self, x_curr: torch.Tensor, d_curr: torch.Tensor, dt: float | torch.Tensor
    ) -> torch.Tensor:
        """Do the integration step.

        Args:
            x_curr: Current state
            d_curr: Current derivative
            dt: Time step

        Returns:
            x_next: Next state
        """
        return x_curr + d_curr * dt

    def _velocity_fn(
        self, x: torch.Tensor, t: torch.Tensor, ctxt_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        model_out = self.model_fn(x, t.expand(x.shape[0]), ctxt_dict)
        return self.prediction.to_velocity(x, model_out, t)

    @torch.no_grad()
    def _step(
        self,
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
        deriv = self._velocity_fn(
            fastsim,
            t=timestep,
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt)
        return fastsim_next, deriv

    @torch.no_grad()
    def _midpoint_step(
        self,
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
        deriv = self._velocity_fn(
            fastsim,
            t=timestep,
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt / 2)
        deriv_next = self._velocity_fn(
            fastsim_next,
            t=(timestep + dt / 2),
            ctxt_dict=ctxt_dict,
        )
        fastsim_next = self._transfer(fastsim, deriv_next, dt)
        return fastsim_next, deriv_next

    @torch.no_grad()
    def _heun_step(
        self,
        fastsim,
        ctxt_dict,
        t_cur,
        t_next,
        is_last=False,
        **kwargs,
    ):
        fastsim_next, deriv_cur = self._step(
            fastsim,
            timestep=t_cur,
            dt=t_next - t_cur,
            ctxt_dict=ctxt_dict,
        )
        deriv_prev = deriv_cur
        if not is_last:
            _, deriv_prime = self._step(
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
        fastsim,
        t_list,
        *,
        ctxt_dict,
        **kwargs,
    ):
        x_2, e_1 = self._step(
            fastsim,
            timestep=t_list[0],
            dt=t_list[1] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        x_3, e_2 = self._step(
            x_2,
            timestep=t_list[1],
            dt=t_list[1] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        x_4, e_3 = self._step(
            x_3,
            timestep=t_list[1],
            dt=t_list[2] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        _, e_4 = self._step(
            x_4,
            timestep=t_list[2],
            dt=t_list[2] - t_list[0],
            ctxt_dict=ctxt_dict,
        )

        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        fastsim_next = self._transfer(fastsim, et, t_list[2] - t_list[0])
        return fastsim_next, et

    def _prepare(
        self,
        target_shape,
        ctxt_dict,
        init_fastsim: torch.Tensor | None = None,
        n_steps: int | None = None,
    ):
        if n_steps is not None and n_steps != self.n_steps:
            self.init_time_steps(n_steps)
        device, dtype = ctxt_dict["ctxt_data"].device, ctxt_dict["ctxt_data"].dtype
        if init_fastsim is None:
            fastsim = self._init_fastsim(target_shape, ctxt_dict.get("fs_mask"), device)
        else:
            fastsim = init_fastsim.to(device)
        t_steps = self.t_steps.to(device).to(dtype)
        return fastsim, t_steps, device, dtype

    @abstractmethod
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass


class EulerSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, device, dtype = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        seq = [fastsim.cpu()]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._step(
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
        if to_cpu:
            return fastsim.cpu()
        return fastsim


class NonSingularSampler(Sampler):
    # From http://arxiv.org/abs/2410.02217 paper
    def __init__(
        self,
        model_fn,
        n_steps,
        prediction: FlowPrediction,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        g_scale=0.0,
        n=1.0,
        m=0.0,
    ):
        super().__init__(
            model_fn=model_fn,
            n_steps=n_steps,
            prediction=prediction,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
        )
        self.g_scale = g_scale
        self.n = n
        self.m = m
        print(f"Using Non-Singular sampler with g_scale={g_scale}, n={n}, m={m}")

    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, device, dtype = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        seq = [fastsim.cpu()]
        g_scale, n, m = self.g_scale, self.n, self.m

        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            tb = t_cur.view(-1, 1, 1)
            v = self._velocity_fn(fastsim, t_cur, ctxt_dict)
            dt = t_next - t_cur

            b = g_scale

            g = b * torch.pow(tb, n / 2) * torch.pow(1.0 - tb, m / 2)

            s_u = -((1.0 - tb) * v + fastsim)

            fr = v - (b**2 * torch.pow(tb, n - 1) * torch.pow(1.0 - tb, m) * s_u / 2.0)

            dbt = torch.sqrt(torch.abs(dt)) * torch.randn_like(fastsim)

            fastsim = fastsim + fr * dt + g * dbt
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            seq = torch.stack(seq)
            return fastsim.cpu(), seq
        if to_cpu:
            return fastsim.cpu()
        return fastsim


class SDESampler(Sampler):
    def __init__(
        self,
        model_fn,
        n_steps,
        prediction: FlowPrediction,
        use_edm_schedule=False,
        heavy_tail=False,
        zero_init_padded=True,
        edm_config: dict | None = None,
        random_seed=None,
        eta: float = 1.0,
        last_step_size: float = 1e-3,
        diffusion_mode: str = "linear",
    ):
        self.eta = eta
        self.last_step_size = last_step_size

        super().__init__(
            model_fn=model_fn,
            n_steps=n_steps,
            prediction=prediction,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
        )
        assert self.reverse_time, (
            "SDE sampler currently only supports reverse time (RCFM)"
        )

        print(
            f"Using SDE sampler with eta={eta}, last_step_size={last_step_size}, diffusion_mode={diffusion_mode}"
        )
        print("Using t_steps:", self.t_steps)
        self.diffusion_mode = diffusion_mode

    def init_time_steps(self, n_steps: int):
        t_start, _ = self.prediction.sampling_time_range()
        self.t_steps = torch.linspace(t_start, self.last_step_size, n_steps + 1)
        self.n_steps = n_steps

    def get_diffusion_coeff(self, t):
        if self.diffusion_mode == "linear":
            return t
        elif self.diffusion_mode == "sdbm":
            return 2 * (t + t**2 / (1 - t))
        elif self.diffusion_mode == "increasing-decreasing":
            return torch.sin(np.pi * t) ** 2
        else:
            raise ValueError(f"Invalid diffusion mode: {self.diffusion_mode}")

    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: Optional[torch.Tensor] = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, device, dtype = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )

        seq = [fastsim.cpu()]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            dt = torch.abs(t_next - t_cur)
            v_theta = self._velocity_fn(
                fastsim,
                t_cur,
                ctxt_dict,
            )
            eps_hat = fastsim + (1.0 - t_cur) * v_theta
            score = -eps_hat / t_cur

            g_t = self.get_diffusion_coeff(t_cur) * self.eta

            drift = v_theta + 0.5 * (g_t**2) * score

            noise = self.distribution.sample(fastsim.shape).to(
                device=device, dtype=dtype
            )
            fastsim = fastsim - drift * dt + g_t * torch.sqrt(dt) * noise

            if save_seq:
                seq.append(fastsim.cpu())

        t_final = torch.tensor(self.last_step_size, device=device, dtype=dtype)
        v_theta = self._velocity_fn(fastsim, t_final, ctxt_dict)
        fastsim = fastsim - v_theta * t_final

        if save_seq:
            seq = torch.stack(seq)
            return fastsim.cpu(), seq
        if to_cpu:
            return fastsim.cpu()
        return fastsim


class MidpointSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, _, _ = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        seq = [fastsim.cpu()]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._midpoint_step(
                fastsim,
                timestep=t_cur,
                dt=t_next - t_cur,
                ctxt_dict=ctxt_dict,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if to_cpu:
            fastsim = fastsim.cpu()
        if save_seq:
            seq = torch.stack(seq)
            return fastsim, seq
        return fastsim


class HeunSampler(Sampler):
    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, _, _ = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._heun_step(
                fastsim,
                t_cur=t_cur,
                t_next=t_next,
                is_last=(i == self.n_steps - 1),
                ctxt_dict=ctxt_dict,
            )
        if to_cpu:
            fastsim = fastsim.cpu()
        return fastsim


class RK4Sampler(Sampler):
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, _, _ = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            fastsim, _ = self._rk4_step(
                fastsim,
                t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                ctxt_dict=ctxt_dict,
            )
        if to_cpu:
            fastsim = fastsim.cpu()
        return fastsim


class PNDMSampler(Sampler):
    def __init__(
        self,
        model_fn,
        n_steps,
        prediction: FlowPrediction,
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
            prediction: Flow path and model-output parameterization
        """
        self.init_step = init_step
        assert self.init_step in ["rk4", "heun", "euler"], "Invalid init_step"
        super().__init__(
            model_fn=model_fn,
            n_steps=n_steps,
            prediction=prediction,
            use_edm_schedule=use_edm_schedule,
            heavy_tail=heavy_tail,
            zero_init_padded=zero_init_padded,
            edm_config=edm_config,
            random_seed=random_seed,
        )

    def _initial_step(
        self,
        fastsim: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        ctxt_dict: dict[str, torch.Tensor],
        is_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.init_step == "rk4":
            return self._rk4_step(
                fastsim,
                t_list=[t_cur, (t_cur + t_next) / 2, t_next],
                ctxt_dict=ctxt_dict,
            )
        if self.init_step == "heun":
            return self._heun_step(
                fastsim,
                t_cur=t_cur,
                t_next=t_next,
                is_last=is_last,
                ctxt_dict=ctxt_dict,
            )
        return self._step(
            fastsim,
            timestep=t_cur,
            dt=t_next - t_cur,
            ctxt_dict=ctxt_dict,
        )

    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        *,
        ctxt_dict,
        save_seq=False,
        init_fastsim: torch.Tensor | None = None,
        to_cpu=True,
        n_steps=None,
        **kwargs,
    ):
        fastsim, t_steps, _, _ = self._prepare(
            target_shape, ctxt_dict, init_fastsim, n_steps
        )
        ets = []
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=self.n_steps
        ):
            if len(ets) > 2:
                deriv_ = self._velocity_fn(
                    fastsim,
                    t=t_cur,
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_)
                deriv = (1 / 24) * (
                    55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]
                )
                fastsim = self._transfer(fastsim, deriv, t_next - t_cur)
                deriv_prev = deriv
            else:
                fastsim, deriv_prev = self._initial_step(
                    fastsim,
                    t_cur=t_cur,
                    t_next=t_next,
                    is_last=(i == self.n_steps - 1),
                    ctxt_dict=ctxt_dict,
                )
                ets.append(deriv_prev)
            if len(ets) > 4:
                ets.pop(0)
        if to_cpu:
            fastsim = fastsim.cpu()
        return fastsim
