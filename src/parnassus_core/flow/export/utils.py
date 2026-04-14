from pathlib import Path
from typing import Iterable, Literal
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.export.model import ExportModelWrapper
from parnassus_core.data.dataset_base import PhysicsDatasetBase

from torch.export import Dim
from argparse import ArgumentParser


EVENT_ONNX_DYNAMIC_AXES = {
    "fs_data": {0: "batch"},
    "timestep": {0: "batch"},
    "mask": {0: "batch"},
    "ctxt_data": {0: "batch"},
    "ctxt_global_data": {0: "batch"},
}

PARTICLE_ONNX_DYNAMIC_AXES = {
    "fs_data": {0: "batch"},
    "timestep": {0: "batch"},
    "mask": {0: "batch"},
    "ctxt_data": {0: "batch"},
    "ctxt_global_data": {0: "batch"},
}

BATCH = Dim("batch", min=1, max=4096)
EVENT_TORCH_DYNAMIC_AXES = {
    "fs_data": {0: BATCH, 1: Dim.STATIC},
    "timestep": {0: BATCH},
    "mask": {0: BATCH, 1: Dim.STATIC},
    "ctxt_data": {0: BATCH, 1: Dim.STATIC, 2: Dim.STATIC},
    "ctxt_global_data": {0: BATCH, 1: Dim.STATIC},
}

PARTICLE_TORCH_DYNAMIC_AXES = {
    "fs_data": {0: BATCH, 1: Dim.STATIC, 2: Dim.STATIC},
    "timestep": {0: BATCH},
    "mask": {0: BATCH, 1: Dim.STATIC, 2: Dim.STATIC},
    "ctxt_data": {0: BATCH, 1: Dim.STATIC, 2: Dim.STATIC},
    "ctxt_global_data": {0: BATCH, 1: Dim.STATIC},
}

DATA_TYPE = dict[str, Tensor]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the exported model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="torch",
        choices=["onnx", "torch"],
        help="Export format: 'onnx' or 'torch'. Default is 'torch'.",
    )
    parser.add_argument(
        "--ds",
        type=str,
        default=None,
        help="Optional path to a dataset file for validation. If not provided,"
        " a sample batch from the training data will be used for validation.",
    )

    args = parser.parse_args()
    return args


def load_model_and_config(
    config_path: str, ckpt_path: str
) -> tuple[FlowLightning, dict]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = FlowLightning.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
    )
    model.net.change_backend("torch")
    model.eval()
    return model, config


def export(
    model: FlowLightning,
    config: dict,
    mode: Literal["onnx", "torch"],
    output_path: str | Path,
    sample_batch: DATA_TYPE,
) -> tuple[ExportModelWrapper, Path]:
    if isinstance(output_path, str):
        output_path = Path(output_path)

    match model_type := config["data"]["train_type"]:
        case "event":
            dynamic_axes = (
                EVENT_ONNX_DYNAMIC_AXES if mode == "onnx" else EVENT_TORCH_DYNAMIC_AXES
            )
        case "particle":
            dynamic_axes = (
                PARTICLE_ONNX_DYNAMIC_AXES
                if mode == "onnx"
                else PARTICLE_TORCH_DYNAMIC_AXES
            )
        case _:
            raise ValueError(f"Unknown data type: {config['data']['train_type']}")

    use_pf_ctxt = False
    if "pf_ctxt_data" in sample_batch:
        if mode == "onnx":
            dynamic_axes["pf_ctxt_data"] = {0: "batch"}
        else:
            dynamic_axes["pf_ctxt_data"] = {0: BATCH, 1: Dim.STATIC, 2: Dim.STATIC}
        use_pf_ctxt = True

    model_wrapper = ExportModelWrapper(
        net=model.net,
        dynamic_axes=dynamic_axes,
        model_type=model_type,
        use_pf_ctxt=use_pf_ctxt,
    )
    model_wrapper.eval()
    input_varnames = [key for key in sample_batch.keys()]
    input_varlist = tuple(sample_batch[name] for name in input_varnames)
    with torch.no_grad():
        if mode == "onnx":
            torch.onnx.export(
                model_wrapper,
                input_varlist,
                input_names=input_varnames,
                dynamic_axes=model_wrapper.dynamic_axes,
                f=output_path,
                opset_version=16,
                report=True,
            )
        elif mode == "torch":
            exported_model = torch.export.export(
                model_wrapper,
                input_varlist,
                dynamic_shapes=model_wrapper.dynamic_axes,
            )
            torch.export.save(exported_model, output_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return model_wrapper, output_path


def validate(
    model_wrapper: ExportModelWrapper,
    output_path: str | Path,
    input_data: DATA_TYPE | Iterable[DATA_TYPE],
    mode: Literal["onnx", "torch"],
    device: torch.device = torch.device("cpu"),
) -> bool:
    if isinstance(output_path, str):
        output_path = Path(output_path)

    if isinstance(input_data, dict):
        input_data = [input_data]
    model_wrapper.to(torch.device("cpu"))
    model_wrapper.eval()
    is_valid = True
    with torch.no_grad():
        if mode == "onnx":
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model, full_check=True)
            # onnx.checker.check_graph(onnx_model.graph)
            providers = ["CPUExecutionProvider"]
            ort_session = ort.InferenceSession(str(output_path), providers=providers)
            input_names = [inp.name for inp in ort_session.get_inputs()]
            print(f"ONNX model input names: {input_names}")
            for batch in input_data:
                ort_inputs = {name: batch[name].cpu().numpy() for name in input_names}
                wrapper_inputs = [batch[name] for name in input_names]
                ort_outs = ort_session.run(None, ort_inputs)
                onnx_output = torch.tensor(ort_outs[0])
                model_output = model_wrapper(*wrapper_inputs)
                if batch["mask"].ndim == 3:
                    particle_mask = batch["mask"][..., 1]
                    model_output = model_output * particle_mask.unsqueeze(-1)
                    onnx_output = onnx_output * particle_mask.unsqueeze(-1)
                if not torch.allclose(model_output, onnx_output, atol=1e-5):
                    abs_error = (model_output - onnx_output).abs().max()
                    print("ONNX model output does not match the original model output.")
                    print(f"Max absolute error: {abs_error}")
                    is_valid = False
        elif mode == "torch":
            loaded_model = torch.export.load(output_path).module()
            loaded_model = torch.compile(loaded_model)
            loaded_model.to(device)
            model_wrapper.to(device)
            model_wrapper.eval()
            for batch in input_data:
                input_varlist = [batch[name].to(device) for name in batch.keys()]

                loaded_output = loaded_model(*input_varlist)
                model_output = model_wrapper(*input_varlist)
                if batch["mask"].ndim == 3:
                    particle_mask = batch["mask"][..., 1].to(device)
                    model_output = model_output * particle_mask.unsqueeze(-1)
                    loaded_output = loaded_output * particle_mask.unsqueeze(-1)
                if not torch.allclose(model_output, loaded_output, atol=1e-5):
                    abs_error = (model_output - loaded_output).abs().max()
                    print(
                        "Torch exported model output does not match the original model output."
                    )
                    print(f"Max absolute error: {abs_error}")
                    is_valid = False
        else:
            model_wrapper.to(torch.device("cpu"))
            if mode == "torch":
                loaded_model.to(torch.device("cpu"))
            raise ValueError(f"Unknown mode: {mode}")

    model_wrapper.to(torch.device("cpu"))
    if mode == "torch":
        loaded_model.to(torch.device("cpu"))
    if is_valid:
        print(f"Model validation successful for {mode} export.")
    return is_valid


def prepare_data(fs_data, ctxt_dict, mode="particle") -> DATA_TYPE:
    timestep = torch.rand(fs_data.shape[0], 1).to(fs_data.dtype)
    if mode == "particle":
        mask = torch.stack([ctxt_dict["ctxt_mask"], ctxt_dict["fs_mask"]], dim=-1)
    else:
        mask = ctxt_dict["ctxt_mask"]
    ctxt_data = ctxt_dict["ctxt_data"]
    ctxt_global_data = ctxt_dict["ctxt_global_data"]

    output: dict[str, Tensor] = {
        "fs_data": fs_data,
        "timestep": timestep,
        "mask": mask,
        "ctxt_data": ctxt_data,
        "ctxt_global_data": ctxt_global_data,
    }
    if "pf_ctxt_data" in ctxt_dict:
        output["pf_ctxt_data"] = ctxt_dict["pf_ctxt_data"]
    return output


def export_and_validate(
    model: FlowLightning,
    config: dict,
    dataset: PhysicsDatasetBase,
    batch_size: int,
    mode: Literal["onnx", "torch"],
    output_path: str | Path,
) -> bool:
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    is_valid = True
    sample_batch = prepare_data(*(next(iter(dl))), mode=dataset.train_type)
    outpath = Path(output_path)
    model_wrapper, outpath = export(
        model,
        config,
        mode,
        outpath,
        sample_batch,
    )
    print(f"Model exported to {outpath.absolute()}")

    # Validate the exported model
    print("Validating the exported model...")
    print("Single batch validation:")
    is_valid = validate(model_wrapper, outpath, sample_batch, mode=mode)

    print("Multiple batch validation:")
    batches = []
    for i, (fs_data, ctxt_dict) in enumerate(dl):
        if i >= 5:
            break
        batch = prepare_data(fs_data, ctxt_dict, mode=dataset.train_type)
        batches.append(batch)

    is_valid = validate(model_wrapper, outpath, batches, mode=mode)

    if torch.cuda.is_available() and mode == "torch":
        print("Multiple batch CUDA validation:")
        is_valid = validate(
            model_wrapper, outpath, batches, mode=mode, device=torch.device("cuda")
        )

    print("Different batch size validation:")
    dl_bs2 = DataLoader(
        dataset,
        batch_size=batch_size // 2,
        shuffle=False,
    )
    sample_batch_bs2 = prepare_data(*(next(iter(dl_bs2))), mode=dataset.train_type)
    is_valid = validate(model_wrapper, outpath, sample_batch_bs2, mode=mode)
    return is_valid
