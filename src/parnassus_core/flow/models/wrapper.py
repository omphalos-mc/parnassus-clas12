from typing import Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from parnassus_core.flow.modules.optimizer import Lion, Muon
from parnassus_core.flow.sampling import (
    DPM_Solver,
    EulerSampler,
    NoiseScheduleFlow,
    PNDMSampler,
)
from parnassus_core.flow.utils.conditional_flow_matcher import FlowMatcher
from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)


class FlowLightning(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        loss: nn.Module,
        fm: FlowMatcher,
        lrs_config: dict,
        optimizer_kwargs: dict,
        optimizer: Literal["AdamW", "Lion", "Muon"] = "AdamW",
        sampler_type: Literal["euler", "pndm", "dpm"] = "euler",
        n_steps: int = 25,
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
        self.change_sampler(sampler_type, n_steps)

    def change_sampler(
        self, sampler_type: Literal["euler", "pndm", "dpm"], n_steps: int = 25
    ):
        self.sampler_type = sampler_type
        self.n_steps = n_steps
        if self.sampler_type == "euler":
            self.sampler = EulerSampler(
                n_steps=self.n_steps, reverse_time=self.FM.flow_type == "rcfm"
            )
        elif self.sampler_type == "pndm":
            self.sampler = PNDMSampler(
                n_steps=self.n_steps,
                reverse_time=self.FM.flow_type == "rcfm",
            )
        elif self.sampler_type == "dpm":

            def model_fn(x, timestep, ctxt_dict):
                return (1 - timestep.view(self.net.time_shape)) * self.net(
                    x, timestep=timestep, ctxt_dict=ctxt_dict
                ) + x

            self.sampler = DPM_Solver(
                model_fn=model_fn, noise_schedule=NoiseScheduleFlow()
            )
        print(f"Using sampler: {self.sampler_type} with {self.n_steps} steps")

    def forward(self, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        pf_data, ctxt_dict = data
        pf_mask = ctxt_dict["fs_mask"]
        t, xt, ut = self.FM.sample(pf_data)
        vt = self.net(
            xt,
            timestep=t,
            ctxt_dict=ctxt_dict,
        )
        loss = self.loss(vt, ut, pf_mask if pf_data.dim() > 2 else None)
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
                method="multistep",
                skip_type="time_uniform_flow",
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
        elif self.optimizer == "muon":
            optimizer = Muon(
                muon_params=self.net.parameters(),
                adamw_params=self.net.parameters(),
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if not self.lrs_config.get("skip_scheduler"):
            if self.lrs_config.get("lr_scheduler_type", "cosine") == "warmup_decay":
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
                    "interval": "step",
                }
            elif self.lrs_config.get("lr_scheduler_type", "cosine") == "cosine":
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
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer,
                    **default_scheduler_dict,
                )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        print("Skipping learning rate scheduler.")
        return optimizer
