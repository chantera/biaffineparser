import os
from collections import UserList, defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch

import utils

T = TypeVar("T")


def _apply(s: str, f: Optional[Callable[[str], str]]) -> str:
    return f(s) if f is not None else s


class Preprocessor:
    serialize_embeddings: bool = False

    def __init__(self):
        self.vocabs: Dict[str, utils.data.Vocab] = {}
        self._embeddings: Optional[torch.Tensor] = None
        self._embed_file: Optional[Union[str, bytes, os.PathLike]] = None

    def build_vocab(
        self,
        file: Union[str, bytes, os.PathLike],
        unknown: str = "<UNK>",
        preprocess: Optional[Callable[[str], str]] = str.lower,
        min_frequency: int = 2,
        cache_dir: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> None:
        def _build_vocabs(file):
            word_counter: Dict[str, int] = defaultdict(int)
            postag_vocab = utils.data.Vocab(unknown)
            deprel_vocab = utils.data.Vocab()
            for tokens in utils.conll.read_conll(file):
                for token in tokens:
                    word_counter[_apply(token["form"], preprocess)] += 1
                    postag_vocab(token["postag"])
                    deprel_vocab(token["deprel"])
            word_iter = (k for k, v in word_counter.items() if v >= min_frequency)
            word_vocab = utils.data.Vocab.fromkeys(word_iter, unknown)
            word_vocab.preprocess = preprocess
            postag_vocab.freeze()
            deprel_vocab.freeze()
            return {"word": word_vocab, "postag": postag_vocab, "deprel": deprel_vocab}

        self.vocabs.update(_wrap_cache(_build_vocabs, file, cache_dir, suffix=".vocab"))
        if "pretrained_word" not in self.vocabs:
            self.vocabs["pretrained_word"] = utils.data.Vocab.fromkeys([], unknown)

    def load_embeddings(
        self,
        file: Union[str, bytes, os.PathLike],
        unknown: str = "<UNK>",
        preprocess: Optional[Callable[[str], str]] = str.lower,
        cache_dir: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> None:
        def _add_entry(token):
            if token not in vocab:
                nonlocal embeddings
                vocab.append(token)
                embeddings = torch.vstack((embeddings, torch.zeros_like(embeddings[0])))

        vocab, embeddings = _wrap_cache(_load_embeddings, file, cache_dir)
        _add_entry(_apply("<ROOT>", preprocess))
        _add_entry(unknown)
        self.vocabs["pretrained_word"] = utils.data.Vocab.fromkeys(vocab, unknown)
        self.vocabs["pretrained_word"].preprocess = preprocess
        self._embeddings = embeddings
        self._embed_file = file

    def transform(
        self, tokens: Iterable[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        words, postags, heads, deprels = zip(
            *[(token["form"], token["postag"], token["head"], token["deprel"]) for token in tokens]
        )
        sample = (
            torch.tensor([self.vocabs["word"][s] for s in words]),
            torch.tensor([self.vocabs["pretrained_word"][s] for s in words]),
            torch.tensor([self.vocabs["postag"][s] for s in postags]),
            torch.tensor(heads),
            torch.tensor([self.vocabs["deprel"][s] for s in deprels]),
        )
        return sample

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if not self.serialize_embeddings:
            state["_embeddings"] = None
        return state

    @property
    def pretrained_word_embeddings(self) -> Optional[torch.Tensor]:
        if self._embeddings is None and self._embed_file is not None:
            v = self.vocabs["pretrained_word"]
            assert v.unknown_id is not None
            self.load_embeddings(self._embed_file, v.lookup(v.unknown_id), v.preprocess)
            assert len(self.vocabs["pretrained_word"]) == len(v)
        return self._embeddings


def _load_embeddings(file):
    embeddings = utils.data.load_embeddings(file)
    return (list(embeddings.keys()), torch.tensor(list(embeddings.values())))


def _wrap_cache(load_fn, file, cache_dir=None, suffix=".cache"):
    if cache_dir is None:
        return load_fn(file)

    basename = os.path.basename(file)
    if not basename:
        raise ValueError(f"Invalid filename: '{file}'")
    cache_file = os.path.join(cache_dir, f"{basename}{suffix}")

    if os.path.exists(cache_file):
        obj = torch.load(cache_file)
    else:
        obj = load_fn(file)
        torch.save(obj, cache_file)
    return obj


def create_dataloader(
    file: Union[str, bytes, os.PathLike],
    preprocessor: Preprocessor,
    device: Optional[torch.device] = None,
    cache_dir: Optional[Union[str, bytes, os.PathLike]] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    dataset = _wrap_cache(
        lambda f: Dataset(map(preprocessor.transform, utils.conll.read_conll(f))), file, cache_dir
    )
    if kwargs.get("batch_sampler") is None:
        kwargs["batch_sampler"] = BucketSampler(
            dataset,
            key=lambda x: len(x[0]),
            batch_size=kwargs.pop("batch_size", 1),
            shuffle=kwargs.pop("shuffle", False),
            drop_last=kwargs.pop("drop_last", False),
            generator=kwargs.get("generator"),
        )
    kwargs.setdefault("collate_fn", lambda batch: collate(batch, device))
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader


class _UserList(UserList, MutableSequence[T]):
    pass


class ListDataset(_UserList[T], torch.utils.data.Dataset):
    pass


class Dataset(ListDataset[Sequence[torch.Tensor]]):
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["data"] = [tuple(attr.tolist() for attr in item) for item in state["data"]]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.data = [tuple(torch.tensor(attr) for attr in item) for item in self.data]


class BucketSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(
        self,
        data_source: Collection,
        key: Callable[[Any], int],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
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


def collate(
    batch: Iterable[Sequence[torch.Tensor]], device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    if device is not None:
        batch = ([col.to(device) for col in row] for row in batch)
    return [list(field) for field in zip(*batch)]
