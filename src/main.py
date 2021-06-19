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
        preprocessor=preprocessor,
        batch_size=batch_size,
        device=device,
    )
    train_dataloader = create_dataloader(train_file, **loader_config, shuffle=True)
    eval_dataloader = None
    if eval_file:
        eval_dataloader = create_dataloader(eval_file, **loader_config, shuffle=False)

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

    trainer = create_trainer(model, lr=lr, epoch=epochs)
    trainer.add_callback(utils.training.PrintCallback(printer=logger.info))
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
        preprocessor=preprocessor,
        batch_size=batch_size,
        device=device,
    )
    eval_dataloader = create_dataloader(eval_file, **loader_config, shuffle=False)

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

    trainer = create_trainer(model)
    trainer.add_callback(utils.training.PrintCallback(printer=logger.info))
    rel_map = {v: k for k, v in preprocessor.vocabs["rel"].mapping.items()}
    trainer.add_callback(EvaluateCallback(eval_file, rel_map, verbose), priority=0)
    with logging_redirect_tqdm(loggers=[logger]):
        trainer.evaluate(eval_dataloader)


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
