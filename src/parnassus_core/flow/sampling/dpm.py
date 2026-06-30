from typing import final

import torch
from tqdm import tqdm

from parnassus_core.flow.utils.prediction import FlowPrediction

# Taken from https://github.com/NVlabs/Sana/tree/main


class NoiseScheduleFlow:
    def __init__(
        self,
        prediction: FlowPrediction,
        schedule="discrete_flow",
    ):
        """Create a wrapper class for the forward SDE (EDM type)."""
        self.total_N = 1000
        self.T = 1 - 1 / self.total_N
        self.t0 = 1 / self.total_N
        self.schedule = schedule  # ['continuous', 'discrete_flow']
        self.prediction = prediction

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        model_t = self.prediction.model_time(t)
        return self.prediction.path.coefficients(model_t, model_t).sigma

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        model_t = self.prediction.model_time(t)
        return self.prediction.path.coefficients(model_t, model_t).alpha

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        low = torch.full_like(lamb, self.t0)
        high = torch.full_like(lamb, self.T)
        for _ in range(64):
            mid = (low + high) / 2
            move_right = self.marginal_lambda(mid) > lamb
            low = torch.where(move_right, mid, low)
            high = torch.where(move_right, high, mid)
        return (low + high) / 2


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


@final
class DPM_Solver:
    def __init__(
        self,
        model_fn,
        prediction: FlowPrediction,
    ):
        """Construct a DPM-Solver."""
        self.model = model_fn
        self.prediction = prediction
        self.noise_schedule = NoiseScheduleFlow(prediction)

    def data_prediction_fn(self, x, t, ctxt_dict):
        """
        Return the data prediction model (with corrector).
        """
        model_t = self.prediction.model_time(t)
        model_out = self.model(
            x,
            model_t.expand(x.shape[0]),
            ctxt_dict,
        )
        return self.prediction.to_data(x, model_out, model_t)

    def model_fn(self, x, t, ctxt_dict):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        return self.data_prediction_fn(x, t, ctxt_dict)

    def get_time_steps(self, t_T, t_0, N, device, shift=1.0):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        timesteps = torch.linspace(t_T, t_0, N + 1, device=device)
        return shift * timesteps / (1 + (shift - 1) * timesteps)

    def denoise_to_zero_fn(self, x, s, ctxt_dict):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s, ctxt_dict)

    def dpm_solver_first_update(
        self, x, s, t, ctxt_dict, model_s=None, return_intermediate=False
    ):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        # dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        # DPM-Solver++ update
        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s, ctxt_dict)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s

        return x_t

    def multistep_dpm_solver_second_update(
        self,
        x,
        model_prev_list,
        t_prev_list,
        t,
    ):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        # DPM-Solver++ update
        phi_1 = torch.expm1(-h)
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
        return x_t

    def multistep_dpm_solver_third_update(
        self,
        x,
        model_prev_list,
        t_prev_list,
        t,
    ):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        # DPM-Solver++ update
        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            + (alpha_t * phi_2) * D1
            - (alpha_t * phi_3) * D2
        )
        return x_t

    def multistep_dpm_solver_update(
        self,
        x,
        model_prev_list,
        t_prev_list,
        t,
        ctxt_dict,
        order,
    ):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(
                x,
                t_prev_list[-1],
                t,
                ctxt_dict,
                model_s=model_prev_list[-1],
            )
        elif order == 2:
            return self.multistep_dpm_solver_second_update(
                x,
                model_prev_list,
                t_prev_list,
                t,
            )
        elif order == 3:
            return self.multistep_dpm_solver_third_update(
                x,
                model_prev_list,
                t_prev_list,
                t,
            )
        else:
            raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    @torch.no_grad()
    @torch.inference_mode()
    def sample(
        self,
        target_shape,
        ctxt_dict,
        n_steps=20,
        t_start=None,
        t_end=None,
        order=2,
        lower_order_final=True,
        denoise_to_zero=False,
        return_intermediate=False,
        flow_shift=1.0,
        to_cpu=True,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================
        Args:
            target_shape: The shape of the target tensor.
            ctxt_dict: A dictionary containing context data.
            n_steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
            to_cpu: A `bool`. Whether to move the output to CPU. Default is `True`.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = self.noise_schedule.t0 if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, (
            "Time range must be positive and lie within the configured schedule"
        )
        x = torch.randn(
            target_shape,
            device=ctxt_dict["ctxt_data"].device,
            dtype=ctxt_dict["ctxt_data"].dtype,
        )
        device = x.device
        intermediates = []
        with torch.no_grad():
            assert n_steps >= order
            timesteps = self.get_time_steps(
                t_T=t_T,
                t_0=t_0,
                N=n_steps,
                device=device,
                shift=flow_shift,
            )
            assert timesteps.shape[0] - 1 == n_steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [self.model_fn(x, t, ctxt_dict)]
            if return_intermediate:
                intermediates.append(x.cpu())
            # Init the first `order` values by lower order multistep DPM-Solver.
            for step in range(1, order):
                t = timesteps[step]
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    t,
                    ctxt_dict,
                    step,
                )
                if return_intermediate:
                    intermediates.append(x.cpu())
                t_prev_list.append(t)
                model_prev_list.append(self.model_fn(x, t, ctxt_dict))
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in tqdm(range(order, n_steps + 1)):
                t = timesteps[step]
                # We only use lower order for steps < 10
                # if lower_order_final and steps < 10:
                if lower_order_final:  # recommended by Shuchen Xue
                    step_order = min(order, n_steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    t,
                    ctxt_dict,
                    step_order,
                )
                if return_intermediate:
                    intermediates.append(x.cpu())
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                # We do not need to evaluate the final model value.
                if step < n_steps:
                    model_prev_list[-1] = self.model_fn(x, t, ctxt_dict)
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t, ctxt_dict)
                if return_intermediate:
                    intermediates.append(x.cpu())
        if to_cpu:
            x = x.cpu()
        if return_intermediate:
            return x, torch.stack(intermediates, dim=0)
        else:
            return x
