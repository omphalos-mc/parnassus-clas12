from typing import Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from parnassus_core.flow.modules.optimizer import Lion
from parnassus_core.flow.sampling import (
    DPM_Solver,
    EulerSampler,
    PNDMSampler,
    SDESampler,
)
from parnassus_core.flow.utils.flow_matcher import FlowMatcher
from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)
from parnassus_core.flow.utils.loss import MSELoss
from parnassus_core.flow.utils.prediction import FlowPrediction, PredictionType


class FlowLightning(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        loss: nn.Module,
        fm: FlowMatcher,
        lrs_config: dict,
        optimizer_kwargs: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        sampler_type: Literal["euler", "sde", "pndm", "dpm"] = "euler",
        n_steps: int = 25,
        ig_loss_weight: float = 0.0,
        ig_sampling_weight: float = 1.0,
        ig_t_min: float = 0.0,
        ig_t_max: float = 1.0,
        sde_eta: float = 1.0,
        sde_last_step_size: float = 1e-4,
        sde_diffusion_mode: str = "linear",
        prediction_type: PredictionType = "velocity",
        loss_target_type: PredictionType | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = name
        self.net = model
        self.loss = loss
        self.FM: FlowMatcher = fm
        self.optimizer = optimizer.lower()
        self.optimizer_kwargs = optimizer_kwargs
        self.lrs_config = lrs_config
        self.prediction_type = prediction_type
        self.loss_target_type = (
            prediction_type if loss_target_type is None else loss_target_type
        )

        self.prediction = FlowPrediction(
            path=self.FM.path,
            prediction_type=prediction_type,
        )
        self.prediction.validate_parameterization(self.loss_target_type)
        print(
            f"Using prediction type: {self.prediction_type}",
            f"with {self.loss_target_type} loss target",
            f"and flow type: {self.FM.flow_type}",
        )

        self.sampler_type = sampler_type
        self.n_steps = n_steps
        self.ig_loss_weight = ig_loss_weight  # Used for internal guidance
        self.ig_sampling_weight = (
            ig_sampling_weight  # Used for internal guidance during sampling
        )
        self.ig_t_min = ig_t_min
        self.ig_t_max = ig_t_max
        self.sde_eta = sde_eta
        self.sde_last_step_size = sde_last_step_size
        self.sde_diffusion_mode = sde_diffusion_mode

        if self.ig_loss_weight > 0.0:
            self.ig_loss = MSELoss(reduction="mean")

        self.change_sampler(
            sampler_type,
            n_steps,
            ig_sampling_weight,
            ig_t_min,
            ig_t_max,
            sde_eta,
            sde_last_step_size=sde_last_step_size,
            sde_diffusion_mode=sde_diffusion_mode,
        )

    def change_sampler(
        self,
        sampler_type: Literal["euler", "sde", "pndm", "dpm"],
        n_steps: int = 25,
        ig_sampling_weight: float = 0.0,
        ig_t_min: float = 0.0,
        ig_t_max: float = 1.0,
        sde_eta: float = 1.0,
        sde_last_step_size: float = 1e-4,
        sde_diffusion_mode: str = "linear",
    ):
        self.sampler_type = sampler_type
        self.n_steps = n_steps
        self.ig_sampling_weight = ig_sampling_weight
        self.ig_t_min = ig_t_min
        self.ig_t_max = ig_t_max
        self.sde_eta = sde_eta
        self.sde_last_step_size = sde_last_step_size
        self.sde_diffusion_mode = sde_diffusion_mode

        def model_fn(x, timestep, ctxt_dict):
            model_out, ig = self.net(
                x,
                timestep=timestep,
                ctxt_dict=ctxt_dict,
            )
            if (
                self.ig_sampling_weight > 1.0
                and ig is not None
                and (self.ig_t_min < timestep[0] < self.ig_t_max)
            ):
                model_out = ig + self.ig_sampling_weight * (model_out - ig)
            return model_out

        if self.sampler_type == "euler":
            self.sampler = EulerSampler(
                model_fn=model_fn,
                n_steps=self.n_steps,
                prediction=self.prediction,
            )
        elif self.sampler_type == "sde":
            self.sampler = SDESampler(
                model_fn=model_fn,
                n_steps=self.n_steps,
                prediction=self.prediction,
                eta=self.sde_eta,
                last_step_size=self.sde_last_step_size,
                diffusion_mode=self.sde_diffusion_mode,
            )
        elif self.sampler_type == "pndm":
            self.sampler = PNDMSampler(
                model_fn=model_fn,
                n_steps=self.n_steps,
                prediction=self.prediction,
            )
        elif self.sampler_type == "dpm":
            self.sampler = DPM_Solver(
                model_fn=model_fn,
                prediction=self.prediction,
            )
        print(f"Using sampler: {self.sampler_type} with {self.n_steps} steps")

    def forward(self, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        pf_data, ctxt_dict = data
        pf_mask = ctxt_dict["fs_mask"]
        flow_sample = self.FM.sample(pf_data)
        model_out, ig = self.net(
            flow_sample.state,
            timestep=flow_sample.time,
            ctxt_dict=ctxt_dict,
        )
        loss_prediction = self.prediction.convert(
            flow_sample.state,
            model_out,
            flow_sample.time,
            self.loss_target_type,
        )
        target = flow_sample.target(self.loss_target_type)
        loss = self.loss(
            loss_prediction,
            target,
            pf_mask if pf_data.dim() > 2 else None,
        )
        if ig is not None and self.ig_loss_weight > 0.0:
            ig_prediction = self.prediction.convert(
                flow_sample.state,
                ig,
                flow_sample.time,
                self.loss_target_type,
            )
            ig_loss = self.ig_loss(
                ig_prediction,
                target,
                pf_mask if pf_data.dim() > 2 else None,
            )
            loss = loss + self.ig_loss_weight * ig_loss
        return loss

    @torch.no_grad()
    def sample(self, fs_shape, ctxt_dict, n_steps=None, to_cpu=True):
        if n_steps is None:
            n_steps = self.n_steps
        return self.sampler.sample(
            target_shape=fs_shape,
            ctxt_dict=ctxt_dict,
            to_cpu=to_cpu,
            n_steps=n_steps,
        )

    def training_step(self, data, batch_idx):
        loss = self.forward(data)
        self.log(
            "train/loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        return loss

    def validation_step(self, data, batch_idx):
        loss = self.forward(data)
        self.log(
            "val/loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

    def test_step(self, data, batch_idx):
        loss = self.forward(data)
        self.log(
            "test/loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if self.lr_schedulers() is not None:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

    def configure_optimizers(self):
        lr = self.optimizer_kwargs["lr"]

        decay = []
        no_decay = []
        for _, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            # Do not apply weight decay to 1D parameters (biases, layernorm weights)
            if param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups,
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "lion":
            optimizer = Lion(
                optim_groups,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        if not self.lrs_config.get("skip_scheduler"):
            lr_scheduler_type = self.lrs_config.get("lr_scheduler_type", "cosine")
            interval = self.lrs_config.get("interval", "epoch")
            if lr_scheduler_type == "warmup_decay":
                default_scheduler_dict = {
                    "warmup_steps": 1000,
                    "decay_steps": 10000,
                    "min_lr": 1e-5,
                }
                default_scheduler_dict.update(
                    self.lrs_config.get("lr_scheduler_kwargs", {})
                )
                print(
                    "Using warmup_decay scheduler with kwargs:", default_scheduler_dict
                )
                scheduler = {
                    "scheduler": WarmupDecayConstLR(
                        optimizer,
                        warmup_steps=int(default_scheduler_dict["warmup_steps"]),
                        decay_steps=int(default_scheduler_dict["decay_steps"]),
                        min_lr=float(default_scheduler_dict["min_lr"]),
                    ),
                    "interval": interval,
                }
            elif lr_scheduler_type == "cosine":
                default_scheduler_dict = {
                    "first_cycle_steps": 10,
                    "warmup_steps": 4,
                    "max_lr": 4 * float(lr),
                    "min_lr": 1e-5,
                    "gamma": 0.8,
                }
                default_scheduler_dict.update(
                    self.lrs_config.get("lr_scheduler_kwargs", {})
                )
                print(
                    "Using cosine annealing warmup restarts scheduler with kwargs:",
                    default_scheduler_dict,
                )
                scheduler = {
                    "scheduler": CosineAnnealingWarmupRestarts(
                        optimizer,
                        **default_scheduler_dict,
                    ),
                    "interval": interval,
                }
            elif lr_scheduler_type == "onecycle":
                lr_scheduler_kwargs = self.lrs_config.get("lr_scheduler_kwargs", {})
                print("Using OneCycleLR scheduler with kwargs:", lr_scheduler_kwargs)
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=lr_scheduler_kwargs["max"],
                    total_steps=self.trainer.estimated_stepping_batches,
                    div_factor=lr_scheduler_kwargs["max"]
                    / lr_scheduler_kwargs["initial"],
                    final_div_factor=lr_scheduler_kwargs["initial"]
                    / lr_scheduler_kwargs["end"],
                    pct_start=float(lr_scheduler_kwargs["pct_start"]),
                    last_epoch=int(lr_scheduler_kwargs.get("last_epoch", -1)),
                )
                scheduler = {
                    "scheduler": sch,
                    "interval": interval,
                }
            elif lr_scheduler_type == "linear_warmup":
                lr_scheduler_kwargs = self.lrs_config.get("lr_scheduler_kwargs", {})
                print("Using linear warmup scheduler with kwargs:", lr_scheduler_kwargs)
                pct_start = float(lr_scheduler_kwargs.get("pct_start", 0.1))
                warmup_steps = int(self.trainer.estimated_stepping_batches * pct_start)
                start_factor = float(lr_scheduler_kwargs.get("start_factor", 0.1))
                sheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=start_factor,
                            end_factor=1.0,
                            total_iters=warmup_steps,
                        ),
                        torch.optim.lr_scheduler.ConstantLR(
                            optimizer, factor=1.0, total_iters=-1
                        ),
                    ],
                    milestones=[warmup_steps],
                )

                scheduler = {
                    "scheduler": sheduler,
                    "interval": interval,
                }
            else:
                raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
            return [optimizer], [scheduler]
        print("Skipping learning rate scheduler.")
        return optimizer
