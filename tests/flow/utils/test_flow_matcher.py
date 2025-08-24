import torch
import pytest

from parnassus_core.flow.utils.conditional_flow_matcher import (
    pad_t_like_x,
    FlowMatcher,
    ConditionalFlowMatcher,
    ReverseConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

torch.manual_seed(42)

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"

FLOW_TYPES = {
    "icfm": ConditionalFlowMatcher,
    "rcfm": ReverseConditionalFlowMatcher,
    "fm": TargetConditionalFlowMatcher,
    "vp": VariancePreservingConditionalFlowMatcher,
}

BS = 2
SEQ_LEN = 8
DIM = 4


@pytest.fixture(params=["event", "particle"])
def input_data(request):
    if request.param == "event":
        shape = (BS, DIM)
    else:
        shape = (BS, SEQ_LEN, DIM)
    data = torch.ones(shape, device=DEVICE)
    return data


class TestFlowMatcher:
    @pytest.fixture
    def fm(self):
        return FlowMatcher(flow_type="icfm", sigma=0.0, time_distribution="uniform")

    @pytest.mark.parametrize("flow_type", FLOW_TYPES.keys())
    def test_flow_init(self, flow_type):
        fm = FlowMatcher(flow_type=flow_type, sigma=0.0, time_distribution="uniform")
        assert isinstance(fm.FM, FLOW_TYPES[flow_type])

    def test_invalid_flow_type(self):
        with pytest.raises(AssertionError):
            FlowMatcher(
                flow_type="invalid_type", sigma=0.0, time_distribution="uniform"
            )

    def test_invalid_time_distribution(self):
        with pytest.raises(AssertionError):
            FlowMatcher(
                flow_type="icfm", sigma=0.0, time_distribution="invalid_distribution"
            )

    @pytest.mark.parametrize("time_distribution", ["uniform", "pow", "lognorm", "mode"])
    def test_time_sampling(self, fm, input_data, time_distribution):
        data = input_data
        fm.time_distribution = time_distribution
        time = fm.sample_time(size=data.shape[0], device=data.device)
        assert time.shape == (data.shape[0],)

    @pytest.mark.parametrize("flow_type", FLOW_TYPES.keys())
    def test_sample(self, flow_type, input_data):
        data = input_data
        fm = FlowMatcher(flow_type=flow_type, sigma=0.0, time_distribution="uniform")
        t, xt, ut = fm.sample(data)
        assert t.shape == (data.shape[0],)
        assert xt.shape == data.shape
        assert ut.shape == data.shape

        match flow_type:
            case "icfm":
                t = pad_t_like_x(t, xt)
                x0 = (xt - t * data) / (1 - t)
                print(x0)
                torch.testing.assert_close(x0, data - ut)
            case "rcfm":
                t = pad_t_like_x(t, xt)
                x0 = (xt - (1 - t) * data) / t
                torch.testing.assert_close(x0, ut + data)
