from pathlib import Path

from parnassus_core.experiments.cms_2011.dataset import CMSDataset
from parnassus_core.flow.export import (
    parse_args,
    load_model_and_config,
    export_and_validate,
)

BATCH_SIZE = 10


def main():
    args = parse_args()
    model, config = load_model_and_config(args.config, args.ckpt)
    model.net.change_backend("torch")
    model.eval()

    ds = CMSDataset(
        config["data"]["train_path"],
        max_particles=400,
        train_type=config["data"]["train_type"],
        sin_cos=True,
        zero_neutral_vtx=True,
        scale_dict=config["data"]["scale_dict"],
        fs_mask_type=config["data"].get("fs_mask_type", "all"),
        var_dict=config["data"]["var_dict"],
        reduce_ds=1000,
        mode="train",
    )
    is_valid = export_and_validate(
        model,
        config,
        ds,
        batch_size=BATCH_SIZE,
        mode=args.mode,
        output_path=Path(args.output),
    )
    if is_valid:
        print("Export and validation completed successfully.")
    else:
        print("Validation failed.")


if __name__ == "__main__":
    main()
