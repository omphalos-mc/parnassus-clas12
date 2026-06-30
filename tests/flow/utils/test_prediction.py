import pytest
import torch

from parnassus_core.flow.sampling import PNDMSampler
from parnassus_core.flow.sampling.dpm import DPM_Solver
from parnassus_core.flow.utils.probability_path import LinearPath, make_probability_path
from parnassus_core.flow.utils.prediction import FlowPrediction


@pytest.mark.parametrize("shape", [(3, 4), (3, 5, 4)])
@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
def test_prediction_conversions(shape, flow_type, prediction_type):
    torch.manual_seed(7)
    data = torch.randn(shape)
    noise = torch.randn(shape)
    t = torch.tensor([0.2, 0.5, 0.8])

    prediction = FlowPrediction(
        path=make_probability_path(flow_type),
        prediction_type=prediction_type,
    )
    coeffs = prediction.path.coefficients(t, data)
    state = coeffs.alpha * noise + coeffs.sigma * data
    velocity = coeffs.alpha_dot * noise + coeffs.sigma_dot * data
    model_out = {
        "velocity": velocity,
        "data": data,
        "noise": noise,
    }[prediction_type]

    torch.testing.assert_close(
        prediction.to_velocity(state, model_out, t),
        velocity,
    )
    torch.testing.assert_close(
        prediction.to_data(state, model_out, t),
        data,
    )
    torch.testing.assert_close(
        prediction.to_noise(state, model_out, t),
        noise,
    )


@pytest.mark.parametrize("shape", [(3, 4), (3, 5, 4)])
@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
@pytest.mark.parametrize("loss_target_type", ["velocity", "data", "noise"])
def test_convert_prediction_for_loss(
    shape,
    flow_type,
    prediction_type,
    loss_target_type,
):
    prediction = FlowPrediction(
        path=make_probability_path(flow_type),
        prediction_type=prediction_type,
    )
    data = torch.randn(shape)
    noise = torch.randn(shape)
    t = torch.tensor([0.2, 0.5, 0.8])
    coeffs = prediction.path.coefficients(t, data)
    state = coeffs.alpha * noise + coeffs.sigma * data
    velocity = coeffs.alpha_dot * noise + coeffs.sigma_dot * data
    values = {
        "velocity": velocity,
        "data": data,
        "noise": noise,
    }

    converted = prediction.convert(
        state,
        values[prediction_type],
        t,
        loss_target_type,
    )
    torch.testing.assert_close(converted, values[loss_target_type])


@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
def test_dpm_converts_model_output_to_data(flow_type, prediction_type):
    prediction = FlowPrediction(
        path=make_probability_path(flow_type),
        prediction_type=prediction_type,
    )
    data = torch.randn(2, 4)
    noise = torch.randn_like(data)
    model_t = torch.tensor(0.35)
    coeffs = prediction.path.coefficients(model_t, data)
    state = coeffs.alpha * noise + coeffs.sigma * data
    velocity = coeffs.alpha_dot * noise + coeffs.sigma_dot * data
    model_out = {
        "velocity": velocity,
        "data": data,
        "noise": noise,
    }[prediction_type]
    seen_t = None

    def model_fn(x, t, ctxt_dict):
        nonlocal seen_t
        seen_t = t
        return model_out

    solver = DPM_Solver(model_fn=model_fn, prediction=prediction)
    solver_t = model_t if prediction.reverse_time else 1 - model_t
    predicted_data = solver.data_prediction_fn(state, solver_t, {})

    torch.testing.assert_close(seen_t, model_t.expand(data.shape[0]))
    torch.testing.assert_close(predicted_data, data)


@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
def test_dpm_follows_exact_probability_path(monkeypatch, flow_type, prediction_type):
    path = make_probability_path(flow_type)
    prediction = FlowPrediction(path=path, prediction_type=prediction_type)
    data = torch.randn(2, 4)
    source = torch.randn_like(data)

    def model_fn(state, model_time, ctxt_dict):
        del ctxt_dict
        coeffs = path.coefficients(model_time, state)
        inferred_source = (state - coeffs.sigma * data) / coeffs.alpha
        if prediction_type == "data":
            return data
        if prediction_type == "noise":
            return inferred_source
        return coeffs.alpha_dot * inferred_source + coeffs.sigma_dot * data

    solver = DPM_Solver(model_fn=model_fn, prediction=prediction)
    start_time = torch.tensor(solver.noise_schedule.T)
    end_time = torch.tensor(solver.noise_schedule.t0)
    initial_state = path.interpolate(
        source,
        data,
        prediction.model_time(start_time),
    )

    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device, dtype: initial_state.to(device=device, dtype=dtype),
    )
    result = solver.sample(
        initial_state.shape,
        {"ctxt_data": initial_state},
        n_steps=4,
        order=2,
    )
    expected = path.interpolate(
        source,
        data,
        prediction.model_time(end_time),
    )

    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("init_step", ["euler", "heun", "rk4"])
def test_pndm_bootstrap_methods(init_step):
    prediction = FlowPrediction(path=LinearPath(), prediction_type="velocity")

    def model_fn(state, time, ctxt_dict):
        del time, ctxt_dict
        return torch.zeros_like(state)

    sampler = PNDMSampler(
        model_fn=model_fn,
        n_steps=4,
        prediction=prediction,
        init_step=init_step,
    )
    ctxt_dict = {
        "fs_mask": None,
        "ctxt_data": torch.zeros(2, 1),
    }

    result = sampler.sample((2, 4), ctxt_dict=ctxt_dict)

    assert result.shape == (2, 4)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("flow_type", "prediction_type", "expected"),
    [
        ("icfm", "velocity", (0.0, 1.0)),
        ("rcfm", "velocity", (1.0, 0.0)),
        ("icfm", "noise", (1e-5, 1.0)),
        ("rcfm", "noise", (1 - 1e-5, 0.0)),
        ("icfm", "data", (0.0, 1 - 1e-5)),
        ("rcfm", "data", (1.0, 1e-5)),
    ],
)
def test_sampling_time_range_avoids_singular_endpoint(
    flow_type, prediction_type, expected
):
    prediction = FlowPrediction(
        path=make_probability_path(flow_type),
        prediction_type=prediction_type,
    )
    assert prediction.sampling_time_range() == pytest.approx(expected)


def test_invalid_prediction_type_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown prediction type"):
        FlowPrediction(path=LinearPath(), prediction_type="invalid")


def test_invalid_conversion_target_is_rejected():
    prediction = FlowPrediction(path=LinearPath(), prediction_type="velocity")

    with pytest.raises(ValueError, match="Unknown prediction type"):
        prediction.convert(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.full((2,), 0.5),
            "invalid",
        )


@pytest.mark.parametrize("prediction_type", ["data", "noise"])
def test_singular_endpoint_conversion_remains_finite(prediction_type):
    prediction = FlowPrediction(
        path=LinearPath(),
        prediction_type=prediction_type,
    )
    time = torch.tensor([0.0, 1.0])
    state = torch.ones(2, 3)
    model_output = torch.ones_like(state)

    velocity = prediction.to_velocity(state, model_output, time)

    assert torch.isfinite(velocity).all()
