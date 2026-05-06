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
from parnassus_core.flow.utils.conditional_flow_matcher import FlowMatcher
from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)
from parnassus_core.flow.utils.loss import MSELoss


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

        if self.sampler_type == "euler":
            self.sampler = EulerSampler(
                n_steps=self.n_steps, reverse_time=self.FM.flow_type == "rcfm"
            )
        elif self.sampler_type == "sde":
            self.sampler = SDESampler(
                n_steps=self.n_steps,
                reverse_time=self.FM.flow_type == "rcfm",
                eta=self.sde_eta,
                last_step_size=self.sde_last_step_size,
                diffusion_mode=self.sde_diffusion_mode,
            )
        elif self.sampler_type == "pndm":
            self.sampler = PNDMSampler(
                n_steps=self.n_steps,
                reverse_time=self.FM.flow_type == "rcfm",
            )
        elif self.sampler_type == "dpm":

            def model_fn(x, timestep, ctxt_dict):
                vt, ig = self.net(
                    x,
                    timestep=timestep,
                    ctxt_dict=ctxt_dict,
                )
                if (
                    self.ig_sampling_weight > 1.0
                    and ig is not None
                    and (self.ig_t_min < timestep[0] < self.ig_t_max)
                ):
                    vt = ig + self.ig_sampling_weight * (vt - ig)
                return (1 - timestep.view(self.net.time_shape)) * vt + x

            self.sampler = DPM_Solver(model_fn=model_fn)
        print(f"Using sampler: {self.sampler_type} with {self.n_steps} steps")

    def forward(self, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        pf_data, ctxt_dict = data
        pf_mask = ctxt_dict["fs_mask"]
        t, xt, ut = self.FM.sample(pf_data)
        vt, ig = self.net(
            xt,
            timestep=t,
            ctxt_dict=ctxt_dict,
        )
        loss = self.loss(vt, ut, pf_mask if pf_data.dim() > 2 else None)
        if ig is not None and self.ig_loss_weight > 0.0:
            ig_loss = self.ig_loss(
                ig,
                ut,
                pf_mask if pf_data.dim() > 2 else None,
            )
            loss = loss + self.ig_loss_weight * ig_loss
        return loss

    @torch.no_grad()
    def sample(self, fs_shape, ctxt_dict, n_steps=None, to_cpu=True):
        if n_steps is None:
            n_steps = self.n_steps
        if self.sampler_type != "dpm":
            return self.sampler.sample(
                self.net,
                pflow_shape=fs_shape,
                ctxt_dict=ctxt_dict,
                to_cpu=to_cpu,
            )
        elif self.sampler_type == "dpm":
            return self.sampler.sample(
                torch.randn(fs_shape, device=ctxt_dict["ctxt_data"].device),
                ctxt_dict=ctxt_dict,
                steps=n_steps,
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
        self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

    def configure_optimizers(self):
        lr = self.optimizer_kwargs["lr"]
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.net.parameters(),
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "lion":
            optimizer = Lion(
                self.net.parameters(),
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        if not self.lrs_config.get("skip_scheduler"):
            lr_scheduler_type = self.lrs_config.get("lr_scheduler_type", "cosine")
            interval = self.lrs_config.get("interval", "epoch")
            if lr_scheduler_type == "warmup_decay":
                print("Using warmup_decay scheduler")
                default_scheduler_dict = {
                    "warmup_steps": 1000,
                    "decay_steps": 10000,
                    "min_lr": 1e-5,
                }
                default_scheduler_dict.update(
                    self.lrs_config.get("lr_scheduler_kwargs", {})
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
                print("Using cosine annealing warmup restarts scheduler")
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
                scheduler = {
                    "scheduler": CosineAnnealingWarmupRestarts(
                        optimizer,
                        **default_scheduler_dict,
                    ),
                    "interval": interval,
                }
            elif lr_scheduler_type == "onecycle":
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lrs_config["max"],
                    total_steps=self.trainer.estimated_stepping_batches,
                    div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                    final_div_factor=self.lrs_config["initial"]
                    / self.lrs_config["end"],
                    pct_start=float(self.lrs_config["pct_start"]),
                    last_epoch=int(self.lrs_config.get("last_epoch", -1)),
                )
                scheduler = {
                    "scheduler": sch,
                    "interval": interval,
                }
            else:
                raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
            return [optimizer], [scheduler]
        print("Skipping learning rate scheduler.")
        return optimizer
