from dataclasses import dataclass
from typing import Literal

import torch

from parnassus_core.flow.utils.probability_path import (
    PathCoefficients,
    ProbabilityPath,
)


PredictionType = Literal["velocity", "data", "noise"]


@dataclass(frozen=True)
class FlowPrediction:
    """Convert model outputs using a shared alpha/sigma probability path."""

    path: ProbabilityPath
    prediction_type: PredictionType = "velocity"
    min_coefficient: float = 1e-5

    def __post_init__(self) -> None:
        self.validate_parameterization(self.prediction_type)

    @property
    def reverse_time(self) -> bool:
        return self.path.reverse_time

    def validate_parameterization(self, prediction_type: PredictionType) -> None:
        if prediction_type not in {"velocity", "data", "noise"}:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

    def convert(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
        t: torch.Tensor,
        target_type: PredictionType,
    ) -> torch.Tensor:
        self.validate_parameterization(target_type)
        if target_type == self.prediction_type:
            return prediction
        if target_type == "velocity":
            return self.to_velocity(state, prediction, t)
        if target_type == "data":
            return self.to_data(state, prediction, t)
        return self.to_noise(state, prediction, t)

    def to_velocity(
        self, state: torch.Tensor, prediction: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "velocity":
            return prediction

        coeffs = self.path.coefficients(t, state)
        if self.prediction_type == "data":
            data = prediction
            source = (state - coeffs.sigma * data) / self._safe(coeffs.alpha)
        else:
            source = prediction
            data = (state - coeffs.alpha * source) / self._safe(coeffs.sigma)
        return coeffs.alpha_dot * source + coeffs.sigma_dot * data

    def to_data(
        self, state: torch.Tensor, prediction: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "data":
            return prediction

        coeffs = self.path.coefficients(t, state)
        if self.prediction_type == "noise":
            return (state - coeffs.alpha * prediction) / self._safe(coeffs.sigma)

        determinant = self._determinant(coeffs)
        return (coeffs.alpha * prediction - coeffs.alpha_dot * state) / self._safe(
            determinant
        )

    def to_noise(
        self, state: torch.Tensor, prediction: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "noise":
            return prediction

        coeffs = self.path.coefficients(t, state)
        if self.prediction_type == "data":
            return (state - coeffs.sigma * prediction) / self._safe(coeffs.alpha)

        determinant = self._determinant(coeffs)
        return (coeffs.sigma_dot * state - coeffs.sigma * prediction) / self._safe(
            determinant
        )

    def model_time(self, solver_time: torch.Tensor) -> torch.Tensor:
        return self.path.model_time(solver_time)

    def sampling_time_range(self) -> tuple[float, float]:
        return self.path.sampling_time_range(
            self.prediction_type,
            self.min_coefficient,
        )

    @staticmethod
    def _determinant(coeffs: PathCoefficients) -> torch.Tensor:
        return coeffs.alpha * coeffs.sigma_dot - coeffs.alpha_dot * coeffs.sigma

    def _safe(self, coefficient: torch.Tensor) -> torch.Tensor:
        sign = torch.where(coefficient < 0, -1, 1)
        return torch.where(
            coefficient.abs() < self.min_coefficient,
            sign * self.min_coefficient,
            coefficient,
        )
