from typing import Any, List

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer


class Smoother:
    def __init__(self, smooth_mode: str, n_smooth: int) -> None:
        self.smooth_mode = smooth_mode
        self.n_smooth = n_smooth

        # Save previous metrics for smoothing
        self.metrics_history = []

        if self.smooth_mode == "ma":
            self.get_smooth_metric = lambda metrics: sum(metrics) / len(metrics)
        else:
            raise ValueError(f"Unsupported smooth_mode: {self.smooth_mode}")

    def __call__(self, metric: float) -> float:
        # Update metrics history
        self.metrics_history.append(metric)
        if len(self.metrics_history) > self.n_smooth:
            self.metrics_history.pop(0)  # Remove oldest metric

        return self.get_smooth_metric(self.metrics_history)


class ReduceLROnPlateauWU(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: List[float] | float = 0,
        eps: float = 1e-8,
        warmup: int = 0,
    ) -> None:
        super().__init__(
            optimizer,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
        )
        self.warmup = warmup
        self.steps = 0

    def step(self, metric: Any) -> None:
        self.steps += 1
        if self.steps > self.warmup:
            return super().step(metric)


class ReduceLROnPlateauSmooth(ReduceLROnPlateauWU):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        smooth_mode: str = "ma",
        n_smooth: int = 10,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: List[float] | float = 0,
        eps: float = 1e-8,
        warmup: int = 0,
    ) -> None:
        super().__init__(
            optimizer,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            warmup,
        )
        self.smoother = Smoother(smooth_mode=smooth_mode, n_smooth=n_smooth)

    def step(self, metric: Any) -> None:
        metric = float(metric)
        smooth_metric = self.smoother(metric)
        return super().step(smooth_metric)


class EarlyStopper:
    def __init__(self, mode="min", patience=5, warmup=0) -> None:
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.best = None
        self.counter = None
        self.steps = None
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self.best = float("inf") if self.mode == "min" else -float("inf")
        self.counter = 0
        self.steps = 0

    def __call__(self, metric: float) -> bool:
        stop = False

        self.steps += 1

        if self.steps < self.warmup:
            return stop

        if self.mode == "min":
            update_best = metric < self.best
        elif self.mode == "max":
            update_best = metric > self.best

        if update_best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                stop = True

        return stop


class EarlyStopperSmooth(EarlyStopper):
    def __init__(
        self,
        mode="min",
        patience=5,
        warmup=0,
        smooth_mode: str = "ma",
        n_smooth: int = 10,
    ) -> None:
        super().__init__(mode, patience, warmup)
        self.smoother = Smoother(smooth_mode=smooth_mode, n_smooth=n_smooth)

    def __call__(self, metric: float) -> bool:
        metric = float(metric)
        smooth_metric = self.smoother(metric)
        return super().__call__(smooth_metric)
