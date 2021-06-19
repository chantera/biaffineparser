import logging
import os
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Iterator, List

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

import models
import utils
from data import Preprocessor
from utils.conll import read_conll

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def train(
    train_file,
    eval_file=None,
    embed_file=None,
    epochs=20,
    batch_size=5000,
    lr=2e-3,
    cuda=False,
    save_dir=None,
    seed=None,
):
    if seed is not None:
        utils.seed_everything(seed)
    device = torch.device("cuda" if cuda else "cpu")

    preprocessor = Preprocessor()
    preprocessor.build_vocab(train_file)
    if embed_file:
        preprocessor.load_embeddings(embed_file)
    loader_config = dict(
        batch_size=batch_size,
        map_fn=preprocessor.transform,
        bucket_key=lambda x: len(x[0]),
        collate_fn=partial(collate, device=device),
    )
    train_dataloader = create_dataloader(read_conll(train_file), **loader_config, shuffle=True)
    eval_dataloader = None
    if eval_file:
        eval_dataloader = create_dataloader(read_conll(eval_file), **loader_config, shuffle=False)

    word_embeddings = preprocessor.pretrained_word_embeddings
    if word_embeddings is not None:
        word_embeddings = torch.tensor(word_embeddings)
    model_config = dict(
        word_vocab_size=len(preprocessor.vocabs["word"]),
        pretrained_word_vocab_size=len(preprocessor.vocabs["word"]),
        postag_vocab_size=len(preprocessor.vocabs["postag"]),
        pretrained_word_embeddings=word_embeddings,
        n_rels=len(preprocessor.vocabs["rel"]),
    )
    model = build_model(**model_config)
    model.to(device)

    trainer = create_trainer(model, step=forward, lr=lr, epoch=epochs)
    if eval_dataloader:
        rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
        trainer.add_callback(EvaluateCallback(eval_file, rel_map), priority=0)
        if save_dir:
            torch.save(preprocessor, os.path.join(save_dir, "preprocessor.pt"))
            trainer.add_callback(
                utils.training.SaveCallback(save_dir, monitor="eval/UAS", mode="max")
            )
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.fit(train_dataloader, eval_dataloader)


def evaluate(
    eval_file, checkpoint_file, preprocessor_file, batch_size=5000, cuda=False, verbose=False
):
    device = torch.device("cuda" if cuda else "cpu")

    preprocessor = torch.load(preprocessor_file)
    loader_config = dict(
        batch_size=batch_size,
        map_fn=preprocessor.transform,
        bucket_key=lambda x: len(x[0]),
        collate_fn=partial(collate, device=device),
    )
    eval_dataloader = create_dataloader(read_conll(eval_file), **loader_config, shuffle=False)

    checkpoint = torch.load(checkpoint_file)
    model_config = dict(
        word_vocab_size=len(preprocessor.vocabs["word"]),
        pretrained_word_vocab_size=len(preprocessor.vocabs["word"]),
        postag_vocab_size=len(preprocessor.vocabs["postag"]),
        n_rels=len(preprocessor.vocabs["rel"]),
    )
    model = build_model(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    trainer = create_trainer(model, step=forward)
    rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
    trainer.add_callback(EvaluateCallback(eval_file, rel_map, verbose), priority=0)
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.evaluate(eval_dataloader)


def create_dataloader(examples, map_fn=None, bucket_key=None, **kwargs):
    if map_fn:
        examples = map(map_fn, examples)
    dataset = ListDataset(examples)
    if kwargs.get("batch_sampler") is None and bucket_key is not None:
        kwargs["batch_sampler"] = BucketSampler(
            dataset,
            key=bucket_key,
            batch_size=kwargs.pop("batch_size", 1),
            shuffle=kwargs.pop("shuffle", False),
            drop_last=kwargs.pop("drop_last", False),
            generator=kwargs.get("generator"),
        )
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader


class ListDataset(list, torch.utils.data.Dataset):
    pass


class BucketSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(
        self,
        data_source,
        key,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator=None,
    ):
        self.data_source = data_source
        self.key = key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        # NOTE: bucketing is applied only one time to fix the number of batches
        self._buckets = list(self._generate_buckets())

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            indices = torch.randperm(len(self._buckets), generator=self.generator)
            return (self._buckets[i] for i in indices)
        return iter(self._buckets)

    def __len__(self) -> int:
        return len(self._buckets)

    def _generate_buckets(self) -> Iterator[List[int]]:
        lengths: Iterator
        lengths = ((i, float(self.key(sample))) for i, sample in enumerate(self.data_source))

        if self.shuffle:
            perturbation = torch.rand(len(self.data_source), generator=self.generator)
            lengths = ((i, length + noise) for (i, length), noise in zip(lengths, perturbation))
            reverse = torch.rand(1, generator=self.generator).item() > 0.5
            lengths = iter(sorted(lengths, key=lambda x: x[1], reverse=reverse))

        bucket: List[int] = []
        accum_len = 0
        for index, length in lengths:
            length = int(length)
            if accum_len + length > self.batch_size:
                yield bucket
                bucket = []
                accum_len = 0
            bucket.append(index)
            accum_len += length
        if not self.drop_last and bucket:
            yield bucket


def collate(batch, device=None):
    batch = ([torch.tensor(col, device=device) for col in row] for row in batch)
    return [list(field) for field in zip(*batch)]


def build_model(**kwargs):
    embeddings = [
        (kwargs.get(f"{name}_vocab_size", 1), kwargs.get(f"{name}_embed_size", 100))
        for name in ["word", "pretrained_word", "postag"]
    ]
    if kwargs.get("pretrained_word_embeddings") is not None:
        embeddings[1] = kwargs["pretrained_word_embeddings"]
    dropout_ratio = kwargs.get("dropout", 0.33)
    encoder = models.BiLSTMEncoder(
        embeddings,
        reduce_embeddings=[0, 1],
        n_lstm_layers=kwargs.get("n_lstm_layers", 3),
        lstm_hidden_size=kwargs.get("lstm_hidden_size", 400),
        embedding_dropout=kwargs.get("embedding_dropout", dropout_ratio),
        lstm_dropout=kwargs.get("lstm_dropout", dropout_ratio),
        recurrent_dropout=kwargs.get("recurrent_dropout", dropout_ratio),
    )
    model = models.BiaffineParser(
        encoder,
        n_rels=kwargs.get("n_rels"),
        arc_mlp_units=kwargs.get("arc_mlp_units", 500),
        rel_mlp_units=kwargs.get("rel_mlp_units", 100),
        arc_mlp_dropout=kwargs.get("arc_mlp_dropout", dropout_ratio),
        rel_mlp_dropout=kwargs.get("rel_mlp_dropout", dropout_ratio),
    )
    return model


def create_trainer(model, **kwargs):
    optimizer = torch.optim.Adam(
        model.parameters(), kwargs.pop("lr", 0.001), betas=(0.9, 0.9), eps=1e-12
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.75 ** (epoch / 5000))
    kwargs.setdefault("max_grad_norm", 5.0)
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_callback(
        ProgressCallback(),
        utils.training.PrintCallback(printer=logger.info),
    )
    trainer.add_metric("arc_loss", "arc_accuracy", "rel_loss", "rel_accuracy")
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
            logger.info(self.result["raw"])

    def _yield_prediction(self):
        for tokens, (arcs, rels) in zip(read_conll(self.gold_file), self._outputs):
            if len(arcs) != len(tokens):
                raise ValueError("heads must be aligned with tokens")
            for token, head, rel in zip(tokens, arcs, rels):
                token.update(head=head, deprel=rel)
            yield tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparser = subparsers.add_parser("train")
    subparser.add_argument("--train_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--eval_file", type=str, default=None, metavar="FILE")
    subparser.add_argument("--embed_file", type=str, default=None, metavar="FILE")
    subparser.add_argument("--epochs", type=int, default=300, metavar="NUM")
    subparser.add_argument("--batch_size", type=int, default=5000, metavar="NUM")
    subparser.add_argument("--lr", type=float, default=2e-3, metavar="VALUE")
    subparser.add_argument("--cuda", action="store_true")
    subparser.add_argument("--save_dir", type=str, default=None, metavar="DIR")
    subparser.add_argument("--seed", type=int, default=None, metavar="VALUE")

    subparser = subparsers.add_parser("evaluate")
    subparser.add_argument("--eval_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--checkpoint_file", "--ckpt", type=str, required=True, metavar="FILE")
    subparser.add_argument(
        "--preprocessor_file", "--proc", type=str, required=True, metavar="FILE"
    )
    subparser.add_argument("--batch_size", type=int, default=5000, metavar="NUM")
    subparser.add_argument("--cuda", action="store_true")
    subparser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        train(
            args.train_file,
            args.eval_file,
            args.embed_file,
            args.epochs,
            args.batch_size,
            args.lr,
            args.cuda,
            args.save_dir,
            args.seed,
        )
    if args.command == "evaluate":
        evaluate(
            args.eval_file,
            args.checkpoint_file,
            args.preprocessor_file,
            args.batch_size,
            args.cuda,
            args.verbose,
        )
    else:
        parser.print_help()
