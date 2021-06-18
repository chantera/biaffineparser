import logging
from functools import partial
from tempfile import NamedTemporaryFile

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
    cache_dir=None,
    refresh_cache=False,
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
        collate_fn=partial(collate, device=device),
    )
    train_dataloader = create_dataloader(read_conll(train_file), **loader_config, shuffle=True)
    eval_dataloader = None
    if eval_file:
        eval_dataloader = create_dataloader(read_conll(eval_file), **loader_config, shuffle=False)

    model_config = dict(
        word_vocab_size=len(preprocessor.vocabs["word"]),
        pretrained_word_vocab_size=len(preprocessor.vocabs["word"]),
        postag_vocab_size=len(preprocessor.vocabs["postag"]),
        pretrained_word_embeddings=preprocessor.pretrained_word_embeddings,
        n_rels=len(preprocessor.vocabs["rel"]),
    )
    model = build_model(**model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.9), eps=1e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.75 ** (epoch / 5000))

    trainer = utils.training.Trainer(model, (optimizer, scheduler), step, epoch=epochs)
    trainer.add_callback(
        ProgressCallback(),
        utils.training.PrintCallback(printer=logger.info),
    )
    if eval_dataloader:
        rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
        trainer.add_callback(EvaluateCallback(eval_file, rel_map), priority=0)

    trainer.add_metric("arc_loss", "arc_accuracy")
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.fit(train_dataloader, eval_dataloader)


def create_dataloader(examples, map_fn=None, **kwargs):
    if map_fn:
        examples = map(map_fn, examples)
    dataset = ListDataset(examples)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader


class ListDataset(list, torch.utils.data.Dataset):
    pass


def collate(batch, device=None):
    batch = ([torch.tensor(col, device=device) for col in row] for row in batch)
    return [list(field) for field in zip(*batch)]


def build_model(**kwargs):
    embeddings = [
        (kwargs.get(f"{name}_vocab_size", 1), kwargs.get(f"{name}_embed_size", 100))
        for name in ["word", "pretrained_word", "postag"]
    ]
    embeddings[1] = kwargs.get("pretrained_word_embeddings") or embeddings[1]
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


def step(model, batch):
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
            result = utils.conll.evaluate(self.gold_file, f.name, self.verbose)
        metrics.update({"eval/UAS": result["UAS"], "eval/LAS": result["LAS"]})
        self._outputs.clear()

    def _yield_prediction(self):
        for tokens, (arcs, rels) in zip(read_conll(self.gold_file), self._outputs):
            if len(arcs) != len(tokens):
                raise ValueError("heads must be aligned with tokens")
            for token, head, rel in zip(tokens, arcs, rels):
                token.update(head=head, deprel=rel)
            yield tokens


def test(model_file, test_file, cuda=False):
    raise NotImplementedError
    """
    context = utils.Saver.load_context(model_file)
    if context.seed is not None:
        utils.set_random_seed(context.seed, device)

    test_dataset = context.loader.load(test_file, train=False, bucketing=True)
    kwargs = dict(context)
    if context.model_config is not None:
        kwargs.update(context.model_config)
    model = _build_parser(**dict(kwargs))
    model.load_state_dict(torch.load(model_file))
    if device >= 0:
        torch.cuda.set_device(device)
        model.cuda()

    pbar = training.listeners.ProgressBar(lambda n: tqdm(total=n))
    pbar.init(len(test_dataset))
    evaluator = Evaluator(model, context.loader.rel_map, test_file, logging.getLogger())
    model.eval()
    for batch in test_dataset.batch(context.batch_size, colwise=True, shuffle=False):
        xs, ts = batch[:-1], batch[-1]
        ys = model.forward(*xs)
        evaluator.on_batch_end({"train": False, "xs": xs, "ys": ys, "ts": ts})
        pbar.update(len(ts))
    evaluator.on_epoch_validate_end({})
    """


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
    subparser.add_argument("--seed", type=int, default=None, metavar="VALUE", help="Random seed")
    subparser.add_argument("--cache_dir", type=str, default=None, metavar="DIR")
    subparser.add_argument("--refresh", "-r", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.add_argument("--model_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--test_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--cuda", action="store_true")

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
            args.cache_dir,
            args.refresh,
        )
    if args.command == "test":
        test(args.model_file, args.test_file, args.cuda)
    else:
        parser.print_help()
