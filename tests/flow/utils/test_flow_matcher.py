import pytest
import torch
import math

from parnassus_core.flow.utils.flow_matcher import FlowMatcher
from parnassus_core.flow.utils.probability_path import (
    LinearPath,
    TrigonometricPath,
    make_probability_path,
)


@pytest.mark.parametrize("shape", [(2, 4), (2, 8, 4)])
@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
def test_path_interpolation_and_velocity(shape, flow_type):
    source = torch.randn(shape)
    data = torch.randn(shape)
    time = torch.tensor([0.25, 0.75])
    path = make_probability_path(flow_type)
    coeffs = path.coefficients(time, data)

    state = path.interpolate(source, data, time)
    velocity = path.velocity(source, data, time)

    torch.testing.assert_close(
        state,
        coeffs.alpha * source + coeffs.sigma * data,
    )
    torch.testing.assert_close(
        velocity,
        coeffs.alpha_dot * source + coeffs.sigma_dot * data,
    )


@pytest.mark.parametrize(
    ("flow_type", "path_type", "reverse_time"),
    [
        ("icfm", LinearPath, False),
        ("rcfm", LinearPath, True),
        ("vp", TrigonometricPath, False),
    ],
)
def test_path_factory(flow_type, path_type, reverse_time):
    path = make_probability_path(flow_type)
    assert isinstance(path, path_type)
    assert path.reverse_time is reverse_time


@pytest.mark.parametrize("time_distribution", ["uniform", "pow", "lognorm", "mode"])
def test_time_sampling(time_distribution):
    matcher = FlowMatcher(
        flow_type="icfm",
        time_distribution=time_distribution,
    )
    time = matcher.sample_time(32, torch.device("cpu"))

    assert time.shape == (32,)
    assert torch.all((0 <= time) & (time <= 1))


def test_power_time_sampling_applies_configured_exponent(monkeypatch):
    uniform = torch.tensor([0.125, 1.0])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device, dtype: uniform.to(device=device, dtype=dtype),
    )
    matcher = FlowMatcher(time_distribution="pow", power_alpha=3)

    sampled = matcher.sample_time(2, torch.device("cpu"))

    torch.testing.assert_close(sampled, torch.tensor([0.5, 1.0]))


def test_lognormal_time_sampling_applies_location_and_scale(monkeypatch):
    normal = torch.tensor([-1.0, 1.0])
    monkeypatch.setattr(
        torch,
        "randn",
        lambda size, device, dtype: normal.to(device=device, dtype=dtype),
    )
    matcher = FlowMatcher(
        time_distribution="lognorm",
        lognorm_mu=0.5,
        lognorm_sigma=2.0,
    )

    sampled = matcher.sample_time(2, torch.device("cpu"))

    torch.testing.assert_close(sampled, torch.sigmoid(0.5 + 2.0 * normal))


def test_mode_time_sampling_applies_configured_shift(monkeypatch):
    uniform = torch.tensor([0.25, 0.75])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, device, dtype: uniform.to(device=device, dtype=dtype),
    )
    matcher = FlowMatcher(time_distribution="mode", mode_s=-0.25)

    sampled = matcher.sample_time(2, torch.device("cpu"))

    expected = (
        1 - uniform + 0.25 * (torch.cos(math.pi / 2 * uniform).square() - 1 + uniform)
    )
    torch.testing.assert_close(sampled, expected)


@pytest.mark.parametrize("shape", [(2, 4), (2, 8, 4)])
@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
def test_flow_matcher_sample(shape, flow_type):
    matcher = FlowMatcher(flow_type=flow_type)
    data = torch.randn(shape)

    sample = matcher.sample(data)

    torch.testing.assert_close(
        sample.state,
        matcher.path.interpolate(sample.source, data, sample.time),
    )
    torch.testing.assert_close(
        sample.velocity,
        matcher.path.velocity(sample.source, data, sample.time),
    )
    time, state, velocity = sample
    assert time is sample.time
    assert state is sample.state
    assert velocity is sample.velocity
    assert sample.target("noise") is sample.source
    assert sample.target("data") is sample.data
    assert sample.target("velocity") is sample.velocity

    with pytest.raises(ValueError, match="Unknown target type"):
        sample.target("invalid")


@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
def test_probability_path_connects_source_to_data(flow_type):
    path = make_probability_path(flow_type)
    source = torch.randn(2, 4)
    data = torch.randn(2, 4)
    start, end = path.time_range

    start_state = path.interpolate(source, data, torch.tensor(start))
    end_state = path.interpolate(source, data, torch.tensor(end))

    torch.testing.assert_close(start_state, source)
    torch.testing.assert_close(end_state, data, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("flow_type", "solver_time", "model_time"),
    [
        ("icfm", 0.2, 0.8),
        ("rcfm", 0.2, 0.2),
        ("vp", 0.2, 0.8),
    ],
)
def test_model_time_maps_dpm_clock_to_probability_path(
    flow_type, solver_time, model_time
):
    path = make_probability_path(flow_type)

    result = path.model_time(torch.tensor(solver_time))

    torch.testing.assert_close(result, torch.tensor(model_time))


def test_invalid_flow_type():
    with pytest.raises(ValueError, match="Unknown flow type"):
        FlowMatcher(flow_type="invalid")


def test_invalid_time_distribution():
    with pytest.raises(ValueError, match="Unknown time distribution"):
        FlowMatcher(flow_type="icfm", time_distribution="invalid")
