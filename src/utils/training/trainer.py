import dataclasses
from typing import Any, Dict, Optional

import torch


class Trainer:
    @dataclasses.dataclass
    class Config:
        epoch: int = 1
        max_steps: Optional[int] = None
        eval_interval: Optional[int] = None
        max_grad_norm: Optional[float] = None

    def __init__(self, model, optimizer, step=None, config=None, **kwargs):
        if not config:
            config = Trainer.Config(**kwargs)
        elif kwargs:
            config = Trainer.Config(**dict(dataclasses.asdict(config), **kwargs))
        self.config = config

        model.zero_grad()
        self.model = model
        if isinstance(optimizer, tuple):
            self.optimizer, self.scheduler = optimizer
        else:
            self.optimizer, self.scheduler = optimizer, None
        self._forward_fn = step
        self._state = {}
        self._should_stop = False

        self._callbacks = _CallbackHandler()
        self._metrics = _MetricsRecorder()
        self.add_callback(_RecordCallback(self._metrics))
        self.add_metric("loss")

    def _init_context(self, dataloader):
        context = Context(
            self,
            self._state.get("epoch"),
            self._state.get("max_epochs"),
            self._state.get("global_step"),
            self._state.get("max_steps"),
            num_batches=len(dataloader),
            num_examples=len(dataloader.dataset),
        )
        return context

    def fit(self, train_dataloader, eval_dataloader=None):
        num_epochs, max_steps = self.config.epoch, self.config.max_steps
        if max_steps is not None:
            num_updates_per_epoch = len(train_dataloader)
            num_epochs = -(-max_steps // num_updates_per_epoch)
        else:
            max_steps = -1

        self._state = {
            "epoch": -1,
            "max_epochs": num_epochs,
            "global_step": 0,
            "max_steps": max_steps,
        }
        self._should_stop = False
        context = self._init_context(train_dataloader)
        context.train = True

        for cb in list(iter(self._callbacks)):
            if isinstance(cb, _EvaluateCallback):
                self.remove_callback(cb)
        if eval_dataloader:
            self.add_callback(_EvaluateCallback(eval_dataloader, self.config.eval_interval))

        self._trigger("on_train_begin", context)
        for epoch in range(num_epochs):
            self._state["epoch"] = context.epoch = epoch
            self._run_loop(context, train_dataloader, self._training_step)

            if self._should_stop:
                break
        self._trigger("on_train_end", context, metrics=self._metrics.asdict())

        self._state.clear()

    def evaluate(self, eval_dataloader):
        self._should_stop = False
        context = self._init_context(eval_dataloader)
        context.train = False

        self._trigger("on_evaluate_begin", context)
        with torch.no_grad():
            self._run_loop(context, eval_dataloader, self._evaluation_step)
        self._trigger("on_evaluate_end", context, metrics=self._metrics.asdict())

    def _run_loop(self, context, dataloader, step_fn):
        self._trigger("on_loop_begin", context)

        for batch_idx, batch in enumerate(dataloader):
            context.batch_index = batch_idx
            self._trigger("on_step_begin", context)
            output = step_fn(batch)
            context.global_step = self._state["global_step"]
            self._trigger("on_step_end", context, output=output)

            if self._should_stop:
                break

        self._trigger("on_loop_end", context, metrics=self._metrics.asdict())

    def _training_step(self, batch):
        self.model.train()
        loss, output = self._forward(batch)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self._state["global_step"] += 1

        if self._state["global_step"] >= self._state["max_steps"] > 0:
            self._should_stop = True

        return output

    def _evaluation_step(self, batch):
        # TODO: refactor
        if not self._state:
            self._state = {
                "epoch": -1,
                "max_epochs": -1,
                "global_step": 0,
                "max_steps": -1,
            }
        self.model.eval()
        _, output = self._forward(batch)
        return output

    def _forward(self, batch):
        if self._forward_fn:
            output = self._forward_fn(self.model, batch)
        else:
            output = self.model(batch)

        if isinstance(output, dict):
            loss = output.pop("loss")
        elif isinstance(output, torch.Tensor):
            loss, output = output, {}
        else:
            raise TypeError(f"invalid output: {type(output).__name__}")

        output["loss"] = loss.detach()
        return loss, output

    def _trigger(self, event, context, **kwargs):
        self._callbacks.notify(event, context, **kwargs)

    def add_callback(self, *callbacks, priority=100):
        for callback in callbacks:
            self._callbacks.add(callback, priority)

    def remove_callback(self, *callbacks):
        for callback in callbacks:
            self._callbacks.remove(callback)

    def add_metric(self, *keys, reduce=None):
        for key in keys:
            self._metrics.register(f"train/{key}", reduce)
            self._metrics.register(f"eval/{key}", reduce)

    def remove_metric(self, *keys):
        for key in keys:
            self._metrics.deregister(f"train/{key}")
            self._metrics.deregister(f"eval/{key}")

    def terminate(self):
        self._should_stop = True


@dataclasses.dataclass
class Context:
    trainer: Trainer
    epoch: int = -1
    max_epochs: int = -1
    global_step: int = -1
    max_steps: int = -1
    train: bool = False
    batch_index: int = -1
    num_batches: int = -1
    num_examples: int = -1


class Callback:
    def on_train_begin(self, context: Context):
        pass

    def on_train_end(self, context: Context, metrics: Dict[str, Any]):
        pass

    def on_evaluate_begin(self, context: Context):
        pass

    def on_evaluate_end(self, context: Context, metrics: Dict[str, Any]):
        pass

    def on_loop_begin(self, context: Context):
        pass

    def on_loop_end(self, context: Context, metrics: Dict[str, Any]):
        pass

    def on_step_begin(self, context: Context):
        pass

    def on_step_end(self, context: Context, output: Dict[Any, Any]):
        pass


class _CallbackHandler:
    def __init__(self):
        self._entries = {}
        self._cache = []
        self._counter = -1

    def __iter__(self):
        if not self._cache:
            self._cache.extend(k for k, _ in sorted(self._entries.items(), key=lambda x: x[1]))
        return iter(self._cache)

    def notify(self, event, *args, **kwargs):
        for callback in self:
            if hasattr(callback, event):
                getattr(callback, event)(*args, **kwargs)

    def add(self, callback, priority=100):
        if callback in self._entries:
            return
        self._counter += 1
        self._entries[callback] = (priority, self._counter)
        self._cache.clear()

    def remove(self, callback):
        if callback in self._entries:
            del self._entries[callback]
            self._cache.clear()


class _EvaluateCallback(Callback):
    def __init__(self, dataloader, interval=None):
        self.dataloader = dataloader
        self.interval = interval

    def on_step_end(self, context, output):
        if not context.train:
            return

        if self.interval is not None and context.global_step % self.interval == 0:
            context.trainer.evaluate(self.dataloader)

    def on_loop_end(self, context, metrics):
        if context.train and self.interval is None:
            context.trainer.evaluate(self.dataloader)


class _RecordCallback(Callback):
    def __init__(self, metrics: "_MetricsRecorder"):
        self.metrics = metrics

    def on_step_end(self, context, output):
        prefix = "train/" if context.train else "eval/"
        for key in self.metrics:
            if not key.startswith(prefix):
                continue
            v = output.get(key.split("/", 1)[1])
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.dim() == 0:
                    v = v.item()
            self.metrics.push(key, v)

    def on_loop_begin(self, context):
        prefix = "train/" if context.train else "eval/"
        for key in self.metrics:
            if key.startswith(prefix):
                self.metrics.clear(key)


class _MetricsRecorder:
    def __init__(self):
        self._entries = {}
        self._reduced = {}

    def __iter__(self):
        return iter(self._entries)

    def register(self, key, reduce_fn=None):
        self._entries[key] = (reduce_fn, [])
        self._reduced.pop(key, None)

    def deregister(self, key):
        del self._entries[key]
        self._reduced.pop(key, None)

    def push(self, key, value):
        self._entries[key][1].append(value)
        self._reduced.pop(key, None)

    def clear(self, key=None):
        if key is not None:
            self._entries[key][1].clear()
            self._reduced.pop(key, None)
        else:
            for key in self:
                self.clear(key)

    def __getitem__(self, key):
        if key in self._reduced:
            return self._reduced[key]
        reduce_fn, val = self._entries[key]
        if not val:
            return None
        val = reduce_fn(val) if reduce_fn else _reduce_specials(key, val)
        self._reduced[key] = val
        return val

    def asdict(self):
        return {key: self[key] for key in self}


def _reduce_specials(key, val):
    if "loss" in key:
        val = sum(val)
    elif "accuracy" in key:
        if not val:
            val = float("nan")
        elif isinstance(val[0], tuple):
            correct, total = 0, 0
            for v in val:
                correct += v[0]
                total += v[1]
            val = correct / total if total > 0 else float("nan")
        else:
            val = sum(val) / len(val)
    return val
