# fmt: off
__all__ = ["ProgressCallback", "PrintCallback", "MonitorCallback", "EarlyStopCallback", "SaveCallback"]  # noqa
# fmt: on

import operator
import os

import torch
from tqdm import tqdm

from utils.training.trainer import Callback  # isort: skip


class ProgressCallback(Callback):
    def __init__(self):
        self.training_pbar = None
        self.evaluation_pbar = None

    def on_train_begin(self, context):
        self._ensure_close(train=True)
        self.training_pbar = tqdm()

    def on_train_end(self, context, metrics):
        self._ensure_close(train=True)

    def on_evaluate_begin(self, context):
        self._ensure_close(eval=True)
        self.evaluation_pbar = tqdm(leave=self.training_pbar is None)

    def on_evaluate_end(self, context, metrics):
        self._ensure_close(eval=True)

    def on_loop_begin(self, context):
        pbar = self.training_pbar if context.train else self.evaluation_pbar
        pbar.reset(context.num_batches)
        if context.train:
            pbar.set_postfix({"epoch": context.epoch})

    def on_step_end(self, context, output):
        pbar = self.training_pbar if context.train else self.evaluation_pbar
        pbar.update(1)

    def _ensure_close(self, train=False, eval=False):
        if train:
            if self.training_pbar is not None:
                self.training_pbar.close()
            self.training_pbar = None
        if eval:
            if self.evaluation_pbar is not None:
                self.evaluation_pbar.close()
            self.evaluation_pbar = None

    def __del__(self):
        self._ensure_close(train=True, eval=True)


class PrintCallback(Callback):
    def __init__(self, printer=None):
        self.printer = printer or tqdm.write

    def on_loop_end(self, context, metrics):
        label = "train" if context.train else "eval"
        loss = metrics[f"{label}/loss"]
        message = f"[{label}] epoch {context.epoch} - loss: {loss:.4f}"

        prefix = label + "/"
        for key, val in metrics.items():
            if not isinstance(val, float) or not key.startswith(prefix):
                continue
            key = key.split("/", 1)[1]
            if key == "loss":
                continue
            message += f", {key}: {val:.4f}"

        self.printer(message)


class MonitorCallback(Callback):
    def __init__(self, monitor="eval/loss", mode="min"):
        self.monitor = monitor
        self.count = 0
        self.mode = mode

        if self.mode == "min":
            self.monitor_op = operator.lt
            self.best = float("inf")
        elif self.mode == "max":
            self.monitor_op = operator.gt
            self.best = float("-inf")
        else:
            raise ValueError(f"invalid mode: {self.mode}")

    def on_evaluate_end(self, context, metrics):
        current_val = metrics[self.monitor]
        if self.monitor_op(current_val, self.best):
            self.best = current_val
            self.count = 0
        else:
            self.count += 1


class EarlyStopCallback(MonitorCallback):
    def __init__(self, monitor="eval/loss", patience=3, mode="min"):
        super().__init__(monitor, mode)
        self.patience = patience

    def on_evaluate_end(self, context, metrics):
        super().on_evaluate_end(context, metrics)
        if self.count >= self.patience:
            context.trainer.terminate()


class SaveCallback(Callback):
    def __init__(self, output_dir, prefix="", mode="latest", monitor=None):
        if mode not in {"latest", "min", "max"}:
            raise ValueError(f"invalid mode: {self.mode}")
        self.output_dir = output_dir
        self.prefix = prefix
        self.monitor = MonitorCallback(monitor, mode) if monitor else None
        self._checkpoints = []

    def on_evaluate_end(self, context, metrics):
        if self.monitor:
            self.monitor.on_evaluate_end(context, metrics)
            if self.monitor.count > 0:
                return

        trainer = context.trainer
        # TODO: add other configuration
        checkpoint = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "trainer_config": trainer.config,
            "trainer_state": trainer._state,
        }
        file = os.path.join(self.output_dir, f"{self.prefix}step-{context.global_step}.ckpt")
        torch.save(checkpoint, file)

        checkpoints = []
        for ckpt_path in self._checkpoints:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        checkpoints.append(file)
        self._checkpoints = checkpoints
