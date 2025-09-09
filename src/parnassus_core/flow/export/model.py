from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor


from torch.export import Dim


class ExportModelWrapper(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        dynamic_axes: dict[str, type[Dim] | dict[int, str]],
        model_type: Literal["event", "particle"] = "event",
        use_pf_ctxt: bool = False,
    ) -> None:
        super().__init__()
        self.net = net
        self.dynamic_axes = dynamic_axes
        self.model_type = model_type
        self.use_pf_ctxt = use_pf_ctxt

    @torch.no_grad()
    def forward(
        self,
        fs_data: Tensor,
        timestep: Tensor,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        pf_ctxt_data: Tensor | None = None,
    ) -> Tensor:  # type: ignore[override]
        """Forward pass through the model.

        The arguments must be passed in the same order they appear in the feature map.
        """

        ctxt_dict: dict[str, torch.Tensor] = {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
        }
        if self.model_type == "event":
            ctxt_dict["ctxt_mask"] = mask
        elif self.model_type == "particle":
            ctxt_dict["ctxt_mask"] = mask[..., 0]
            ctxt_dict["fs_mask"] = mask[..., 1]
        if self.use_pf_ctxt:
            ctxt_dict["pf_ctxt_data"] = pf_ctxt_data
        outputs = self.net(fs_data, timestep, ctxt_dict)
        return outputs
