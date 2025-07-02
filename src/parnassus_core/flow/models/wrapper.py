import torch
from lightning.pytorch import LightningModule

import numpy as np
from scipy.stats import norm

from parnassus_core.flow.utils.loss import MSELoss, MSEAndDirectionLoss, Set2SetLoss
from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)
from parnassus_core.flow.models.part_net import ParticleNetwork
from parnassus_core.flow.models.evt_net import EventNetwork
from parnassus_core.flow.models.imp_net import ImpactNetwork
from parnassus_core.flow.modules.optimizer import Lion, Muon
from parnassus_core.flow.sampling import DPM_Solver, NoiseScheduleFlow, EulerSampler
from parnassus_core.flow.utils.conditional_flow_matcher import (
    ConditionalFlowMatcher,
    ReverseConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)


def sample_from_logit_normal(mu, sigma, size):
    samples = norm.rvs(loc=mu, scale=sigma, size=size)
    samples = 1 / (1 + np.exp(-samples))
    return samples


def sample_from_mode(s, size):
    t = np.random.rand(size)
    return 1 - t - s * (np.cos(np.pi / 2 * t) ** 2 - 1 + t)


class FlowLightning(LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config

        self.model_type = config["model_type"]
        if self.model_type == "particle_network":
            self.net = ParticleNetwork(config["model"])
            self.prediction_loss = Set2SetLoss()
            self.time_shape = (-1, 1, 1)  # (batch_size, 1, 1)
        elif self.model_type == "event_network":
            self.net = EventNetwork(config["model"])
            self.prediction_loss = MSELoss()
            self.time_shape = (-1, 1)
        elif self.model_type == "impact_network":
            self.net = ImpactNetwork(config["model"])
            self.prediction_loss = MSELoss()
            self.time_shape = (-1, 1, 1)
        self.flow_type = config["flow_type"]
        self.loss_type = config["loss_type"]
        self.sampler_type = config["sampler_type"]
        self.n_steps = int(config["n_steps"])

        if self.loss_type == "mse":
            self.loss = MSELoss()
        elif self.loss_type == "mse_and_direction":
            self.loss = MSEAndDirectionLoss(**config.get("loss_kwargs", {}))
        else:
            raise NotImplementedError(
                f"Unknown loss type {self.loss_type}, must be one of ['mse', 'set2set']"
            )

        self.sigma = config["sigma"]
        self.time_pow = config.get("time_pow", False)
        self.time_lognorm = config.get("time_lognorm", False)
        self.time_mode = config.get("time_mode", False)

        if self.time_pow + self.time_lognorm + self.time_mode > 1:
            raise ValueError(
                "Cannot have more than one of time_pow, time_lognorm, time_mode"
            )
        print("Using time_pow:", self.time_pow)
        print("Using time_lognorm:", self.time_lognorm)
        print("Using time_mode:", self.time_mode)

        self.opt = config.get("opt", "adamw").lower()
        if self.opt not in ["adamw", "lion", "muon"]:
            raise NotImplementedError(
                f"Unknown optimizer {self.opt}, must be one of ['adamw', 'lion', 'muon']"
            )

        if self.flow_type == "icfm":
            self.FM = ConditionalFlowMatcher(sigma=self.sigma)
        elif self.flow_type == "rcfm":
            self.FM = ReverseConditionalFlowMatcher(sigma=self.sigma)
        elif self.flow_type == "fm":
            self.FM = TargetConditionalFlowMatcher(sigma=self.sigma)
        elif self.flow_type == "vp":
            self.FM = VariancePreservingConditionalFlowMatcher(sigma=self.sigma)
        else:
            raise NotImplementedError(
                f"Unknown model {self.flow_type}, must be one of ['icfm', 'rcfm', 'fm', 'vp']"
            )
        if self.sampler_type == "euler":
            self.sampler = EulerSampler(
                n_steps=self.n_steps, reverse_time=self.flow_type == "rcfm"
            )
        elif self.sampler_type == "dpm":

            def model_fn(x, timestep, ctxt_dict):
                return (1 - timestep.view(self.time_shape)) * self.net(
                    x, timestep=timestep, ctxt_dict=ctxt_dict
                ) + x

            self.sampler = DPM_Solver(
                model_fn=model_fn, noise_schedule=NoiseScheduleFlow()
            )

    def forward(self, data):
        pf_data, ctxt_dict = data
        pf_mask = ctxt_dict["fs_mask"] if self.model_type != "event_network" else None
        x0 = torch.randn_like(pf_data)
        if self.time_pow:
            t = torch.FloatTensor(np.random.power(3, size=pf_data.shape[0])).to(
                pf_data.device
            )
        elif self.time_lognorm:
            t = torch.FloatTensor(
                sample_from_logit_normal(0, 1, size=pf_data.shape[0])
            ).to(pf_data.device)
        elif self.time_mode:
            t = torch.FloatTensor(sample_from_mode(-0.54, size=pf_data.shape[0])).to(
                pf_data.device
            )
        else:
            t = None
        t, xt, ut, _ = self.FM.sample_location_and_conditional_flow(
            x0, pf_data, t=t, return_noise=True
        )

        vt = self.net(
            xt,
            timestep=t,
            ctxt_dict=ctxt_dict,
        )
        loss = self.loss(ut, vt, pf_mask)
        return loss

    @torch.no_grad()
    def sample(self, fs_shape, ctxt_dict, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps
        if self.sampler_type == "euler":
            return self.sampler.sample(
                self.net,
                pflow_shape=fs_shape,
                ctxt_dict=ctxt_dict,
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
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)
            self.lr_schedulers().step(epoch=self.current_epoch)

    def configure_optimizers(self):
        self.config["optimizer_kwargs"]["lr"] = float(
            self.config["optimizer_kwargs"]["lr"]
        )
        if self.opt == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                **self.config["optimizer_kwargs"],
            )
        elif self.opt == "lion":
            optimizer = Lion(
                self.parameters(),
                **self.config["optimizer_kwargs"],
            )
        elif self.opt == "muon":
            optimizer = Muon(
                muon_params=self.parameters(),
                adamw_params=self.parameters(),
                **self.config["optimizer_kwargs"],
            )
        if self.config["lr_scheduler"] is False:
            return optimizer
        else:
            print("\nscheduler exists!\n")
            if self.config.get("lr_scheduler_type", "cosine") == "warmup_decay":
                default_scheduler_dict = {
                    "warmup_steps": 1000,
                    "decay_steps": 10000,
                    "min_lr": 1e-5,
                }
                default_scheduler_dict.update(self.config.get("lr_scheduler_dict", {}))
                scheduler = {
                    "scheduler": WarmupDecayConstLR(
                        optimizer,
                        warmup_steps=int(default_scheduler_dict["warmup_steps"]),
                        decay_steps=int(default_scheduler_dict["decay_steps"]),
                        min_lr=float(default_scheduler_dict["min_lr"]),
                    ),
                    "interval": "step",
                }
            elif self.config.get("lr_scheduler_type", "cosine") == "cosine":
                default_scheduler_dict = {
                    "first_cycle_steps": 10,
                    "warmup_steps": 4,
                    "max_lr": 4 * float(self.config["learningrate"]),
                    "min_lr": 1e-5,
                    "gamma": 0.8,
                }
                default_scheduler_dict.update(self.config.get("lr_scheduler_dict", {}))
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer,
                    **default_scheduler_dict,
                )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
