from tempfile import NamedTemporaryFile

import torch
from tqdm import tqdm

import utils


def create_trainer(model, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), kwargs.pop("lr", 0.001), betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.75 ** (step / 5000))
    kwargs.setdefault("max_grad_norm", 5.0)
    kwargs.setdefault("step", forward)
    # NOTE: `scheduler.step()` is called every iteration in the trainer
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_metric("head_loss", "head_accuracy", "deprel_loss", "deprel_accuracy")
    trainer.add_callback(ProgressCallback())
    return trainer


def forward(model, batch):
    *xs, heads, deprels = batch
    logits_head, logits_deprel, lengths = model(*xs)
    result = model.compute_metrics(logits_head, logits_deprel, heads, deprels)
    if logits_deprel is None:
        result.update(deprel_loss=float("nan"), deprel_accuracy=float("nan"))
    if not model.training:
        predicted_heads, predicted_deprels = model.decode(logits_head, logits_deprel, lengths)
        result.update(heads=predicted_heads, deprels=predicted_deprels, lengths=lengths)
    return result


class ProgressCallback(utils.training.ProgressCallback):
    def on_step_end(self, context, **kwargs):
        super().on_step_end(context, **kwargs)
        if context.train:
            loss = kwargs["output"]["loss"].item()
            correct, total = kwargs["output"]["head_accuracy"]
            accuracy = correct / total if total > 0 else float("nan")
            pbar_dict = {
                "epoch": context.epoch,
                "loss": f"{loss:.4f}",
                "accuracy": f"{accuracy:.4f}",
            }
            self.training_pbar.set_postfix(pbar_dict)


class EvaluateCallback(utils.training.Callback):
    printer = tqdm.write

    def __init__(self, gold_file, deprel_map, verbose=False):
        self.gold_file = gold_file
        self.deprel_map = deprel_map
        self.verbose = verbose
        self.result = {}
        self._outputs = []

    def on_step_end(self, context, output):
        if context.train:
            return
        heads, deprels, lengths = output["heads"], output["deprels"], output["lengths"]
        assert heads.size(0) == len(lengths)
        heads = (idxs[:n] for idxs, n in zip(heads.tolist(), lengths))
        if deprels is not None:
            deprel_map = self.deprel_map
            deprels = (
                [deprel_map[idx] for idx in idxs[:n]] for idxs, n in zip(deprels.tolist(), lengths)
            )
        else:
            deprels = ([None] * n for n in lengths)
        self._outputs.extend(zip(heads, deprels))

    def on_loop_end(self, context, metrics):
        if context.train:
            metrics.update({"train/UAS": float("nan"), "train/LAS": float("nan")})
            return
        with NamedTemporaryFile(mode="w") as f:
            utils.conll.dump_conll(self._yield_prediction(), f)
            self.result.update(utils.conll.evaluate(self.gold_file, f.name, self.verbose))
        metrics.update({"eval/UAS": self.result["UAS"], "eval/LAS": self.result["LAS"]})
        self._outputs.clear()

    def on_evaluate_end(self, context, metrics):
        metrics.update({"eval/UAS": self.result["UAS"], "eval/LAS": self.result["LAS"]})
        if self.verbose:
            self.printer(self.result["raw"])

    def _yield_prediction(self):
        for tokens, (heads, deprels) in zip(utils.conll.read_conll(self.gold_file), self._outputs):
            if len(heads) != len(tokens):
                raise ValueError("heads must be aligned with tokens")
            for token, head, deprel in zip(tokens, heads, deprels):
                token.update(head=head, deprel=deprel)
            yield tokens
