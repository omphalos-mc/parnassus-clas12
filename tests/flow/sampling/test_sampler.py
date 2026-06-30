import pytest
import torch

from parnassus_core.flow.sampling import EulerSampler, PNDMSampler, SDESampler
from parnassus_core.flow.utils.prediction import FlowPrediction
from parnassus_core.flow.utils.probability_path import LinearPath


def constant_velocity(value):
    def model_fn(state, time, ctxt_dict):
        del time, ctxt_dict
        return torch.full_like(state, value)

    return model_fn


def context(batch_size=2):
    return {"ctxt_data": torch.zeros(batch_size, 1)}


@pytest.mark.parametrize(
    ("reverse", "expected"),
    [(False, 1.0), (True, -1.0)],
)
def test_euler_integrates_constant_velocity_over_path_time(reverse, expected):
    prediction = FlowPrediction(
        path=LinearPath(reverse=reverse),
        prediction_type="velocity",
    )
    sampler = EulerSampler(
        model_fn=constant_velocity(1.0),
        n_steps=4,
        prediction=prediction,
    )
    initial = torch.zeros(2, 3)

    result = sampler.sample(
        initial.shape,
        ctxt_dict=context(),
        init_fastsim=initial,
    )

    torch.testing.assert_close(result, torch.full_like(initial, expected))


def test_euler_save_seq_contains_initial_and_every_integration_step():
    sampler = EulerSampler(
        model_fn=constant_velocity(2.0),
        n_steps=4,
        prediction=FlowPrediction(LinearPath(), "velocity"),
    )
    initial = torch.zeros(2, 3)

    result, sequence = sampler.sample(
        initial.shape,
        ctxt_dict=context(),
        init_fastsim=initial,
        save_seq=True,
    )

    assert sequence.shape == (5, *initial.shape)
    torch.testing.assert_close(sequence[0], initial)
    torch.testing.assert_close(sequence[-1], result)
    torch.testing.assert_close(result, torch.full_like(initial, 2.0))


def test_runtime_n_steps_rebuilds_schedule_and_controls_model_calls():
    calls = []

    def model_fn(state, time, ctxt_dict):
        del ctxt_dict
        calls.append(time.clone())
        return torch.zeros_like(state)

    sampler = EulerSampler(
        model_fn=model_fn,
        n_steps=2,
        prediction=FlowPrediction(LinearPath(), "velocity"),
    )

    sampler.sample(
        (2, 3),
        ctxt_dict=context(),
        init_fastsim=torch.zeros(2, 3),
        n_steps=5,
    )

    assert sampler.n_steps == 5
    assert sampler.t_steps.shape == (6,)
    assert len(calls) == 5


def test_reset_replays_seeded_initial_noise():
    sampler = EulerSampler(
        model_fn=constant_velocity(0.0),
        n_steps=1,
        prediction=FlowPrediction(LinearPath(), "velocity"),
        random_seed=17,
    )

    first = sampler.sample((2, 3), ctxt_dict=context())
    sampler.reset()
    replay = sampler.sample((2, 3), ctxt_dict=context())

    torch.testing.assert_close(replay, first)


@pytest.mark.parametrize("init_step", ["euler", "heun", "rk4"])
def test_pndm_bootstrap_methods_preserve_constant_velocity(init_step):
    sampler = PNDMSampler(
        model_fn=constant_velocity(1.5),
        n_steps=4,
        prediction=FlowPrediction(LinearPath(), "velocity"),
        init_step=init_step,
    )
    initial = torch.zeros(2, 3)

    result = sampler.sample(
        initial.shape,
        ctxt_dict=context(),
        init_fastsim=initial,
    )

    torch.testing.assert_close(result, torch.full_like(initial, 1.5))


def test_sde_requires_reverse_time_path():
    prediction = FlowPrediction(LinearPath(), "velocity")

    with pytest.raises(AssertionError, match="only supports reverse time"):
        SDESampler(
            model_fn=constant_velocity(0.0),
            n_steps=4,
            prediction=prediction,
        )


def test_sde_with_zero_diffusion_integrates_deterministic_velocity():
    sampler = SDESampler(
        model_fn=constant_velocity(1.0),
        n_steps=4,
        prediction=FlowPrediction(LinearPath(reverse=True), "velocity"),
        eta=0.0,
        last_step_size=0.1,
    )
    initial = torch.zeros(2, 3)

    result = sampler.sample(
        initial.shape,
        ctxt_dict=context(),
        init_fastsim=initial,
    )

    torch.testing.assert_close(result, torch.full_like(initial, -1.0))


@pytest.mark.parametrize(
    ("mode", "time", "expected"),
    [
        ("linear", 0.25, 0.25),
        ("sdbm", 0.25, 2 * (0.25 + 0.25**2 / 0.75)),
        ("increasing-decreasing", 0.5, 1.0),
    ],
)
def test_sde_diffusion_coefficients(mode, time, expected):
    sampler = SDESampler(
        model_fn=constant_velocity(0.0),
        n_steps=2,
        prediction=FlowPrediction(LinearPath(reverse=True), "velocity"),
        diffusion_mode=mode,
    )

    coefficient = sampler.get_diffusion_coeff(torch.tensor(time))

    torch.testing.assert_close(coefficient, torch.tensor(expected))


def test_sde_rejects_unknown_diffusion_mode():
    sampler = SDESampler(
        model_fn=constant_velocity(0.0),
        n_steps=2,
        prediction=FlowPrediction(LinearPath(reverse=True), "velocity"),
        diffusion_mode="unknown",
    )

    with pytest.raises(ValueError, match="Invalid diffusion mode"):
        sampler.get_diffusion_coeff(torch.tensor(0.5))
