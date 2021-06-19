import logging
import os

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

import utils
from data import Preprocessor, create_dataloader
from models import build_model
from training import EvaluateCallback, create_trainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(command=train)
    subparser.add_argument("--train_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--eval_file", type=str, default=None, metavar="FILE")
    subparser.add_argument("--embed_file", type=str, default=None, metavar="FILE")
    subparser.add_argument("--max_steps", type=int, default=50000, metavar="NUM")
    subparser.add_argument("--eval_interval", type=int, default=100, metavar="NUM")
    subparser.add_argument("--batch_size", type=int, default=5000, metavar="NUM")
    subparser.add_argument("--lr", type=float, default=2e-3, metavar="VALUE")
    subparser.add_argument("--cuda", action="store_true")
    subparser.add_argument("--save_dir", type=str, default=None, metavar="DIR")
    subparser.add_argument("--seed", type=int, default=None, metavar="VALUE")

    subparser = subparsers.add_parser("evaluate")
    subparser.set_defaults(command=evaluate)
    subparser.add_argument("--eval_file", type=str, required=True, metavar="FILE")
    subparser.add_argument("--checkpoint_file", "--ckpt", type=str, required=True, metavar="FILE")
    subparser.add_argument(
        "--preprocessor_file", "--proc", type=str, required=True, metavar="FILE"
    )
    subparser.add_argument("--batch_size", type=int, default=5000, metavar="NUM")
    subparser.add_argument("--cuda", action="store_true")
    subparser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    args.command(args)


def train(args):
    if args.seed is not None:
        utils.random.seed_everything(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    preprocessor = Preprocessor()
    preprocessor.build_vocab(args.train_file)
    if args.embed_file:
        preprocessor.load_embeddings(args.embed_file)
    loader_config = dict(
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        device=device,
    )
    train_dataloader = create_dataloader(args.train_file, **loader_config, shuffle=True)
    eval_dataloader = None
    if args.eval_file:
        eval_dataloader = create_dataloader(args.eval_file, **loader_config, shuffle=False)

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

    trainer = create_trainer(
        model, lr=args.lr, max_steps=args.max_steps, eval_interval=args.eval_interval
    )
    trainer.add_callback(utils.training.PrintCallback(printer=logger.info))
    if eval_dataloader:
        rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
        trainer.add_callback(EvaluateCallback(args.eval_file, rel_map), priority=0)
        if args.save_dir:
            torch.save(preprocessor, os.path.join(args.save_dir, "preprocessor.pt"))
            trainer.add_callback(
                utils.training.SaveCallback(args.save_dir, monitor="eval/UAS", mode="max")
            )
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.fit(train_dataloader, eval_dataloader)


def evaluate(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    preprocessor = torch.load(args.preprocessor_file)
    loader_config = dict(
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        device=device,
    )
    eval_dataloader = create_dataloader(args.eval_file, **loader_config, shuffle=False)

    checkpoint = torch.load(args.checkpoint_file)
    model_config = dict(
        word_vocab_size=len(preprocessor.vocabs["word"]),
        pretrained_word_vocab_size=len(preprocessor.vocabs["word"]),
        postag_vocab_size=len(preprocessor.vocabs["postag"]),
        n_rels=len(preprocessor.vocabs["rel"]),
    )
    model = build_model(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    trainer = create_trainer(model)
    trainer.add_callback(utils.training.PrintCallback(printer=logger.info))
    rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
    trainer.add_callback(EvaluateCallback(args.eval_file, rel_map, args.verbose), priority=0)
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.evaluate(eval_dataloader)


if __name__ == "__main__":
    main()
