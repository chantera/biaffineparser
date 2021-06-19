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
    trainer.add_metric("arc_loss", "arc_accuracy", "rel_loss", "rel_accuracy")
    trainer.add_callback(ProgressCallback())
    return trainer


def forward(model, batch):
    *xs, heads, rels = batch
    logits_arc, logits_rel, lengths = model(*xs)
    result = model.compute_metrics(logits_arc, logits_rel, heads, rels)
    if not model.training:
        arcs, rels = model.decode(logits_arc, logits_rel, lengths)
        result.update(arcs=arcs, rels=rels, lengths=lengths)
    return result


class ProgressCallback(utils.training.ProgressCallback):
    def on_step_end(self, context, **kwargs):
        super().on_step_end(context, **kwargs)
        if context.train:
            loss = kwargs["output"]["loss"].item()
            correct, total = kwargs["output"]["arc_accuracy"]
            accuracy = correct / total if total > 0 else float("nan")
            pbar_dict = {
                "epoch": context.epoch,
                "loss": f"{loss:.4f}",
                "accuracy": f"{accuracy:.4f}",
            }
            self.training_pbar.set_postfix(pbar_dict)


class EvaluateCallback(utils.training.Callback):
    printer = tqdm.write

    def __init__(self, gold_file, rel_map, verbose=False):
        self.gold_file = gold_file
        self.rel_map = rel_map
        self.verbose = verbose
        self.result = {}
        self._outputs = []

    def on_step_end(self, context, output):
        if context.train:
            return
        arcs, rels, lengths = output["arcs"], output["rels"], output["lengths"]
        assert arcs.size(0) == len(lengths)
        arcs = (idxs[:n] for idxs, n in zip(arcs.tolist(), lengths))
        if rels is not None:
            rel_map = self.rel_map
            rels = ([rel_map[idx] for idx in idxs[:n]] for idxs, n in zip(rels.tolist(), lengths))
        else:
            rels = ([None] * n for n in lengths)
        self._outputs.extend(zip(arcs, rels))

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
        for tokens, (arcs, rels) in zip(utils.conll.read_conll(self.gold_file), self._outputs):
            if len(arcs) != len(tokens):
                raise ValueError("heads must be aligned with tokens")
            for token, head, rel in zip(tokens, arcs, rels):
                token.update(head=head, deprel=rel)
            yield tokens
