from collections import Counter
from typing import Iterator, List

import torch

import utils


class Preprocessor:
    serialize_embeddings = False

    def __init__(self):
        self.vocabs = {}
        self._embeddings = None
        self._embed_file = None

    def build_vocab(self, file, unknown="<UNK>", preprocess=str.lower, min_frequency=2):
        word_set, postag_set, rel_set = set(), set(), set()
        word_counter = Counter()
        for tokens in utils.conll.read_conll(file):
            words, postags, rels = zip(
                *[
                    (preprocess(token["form"]), token["postag"], token["deprel"])
                    for token in tokens
                ]
            )
            word_counter.update(words)
            postag_set.update(postags)
            rel_set.update(rels)
        for word, count in word_counter.most_common():
            if count < min_frequency:
                break
            word_set.add(word)

        self.vocabs["word"] = utils.data.Vocab.fromkeys(word_set, unknown)
        self.vocabs["word"].preprocess = preprocess
        self.vocabs["postag"] = utils.data.Vocab.fromkeys(postag_set, unknown)
        self.vocabs["rel"] = utils.data.Vocab.fromkeys(rel_set)
        if "pretrained_word" not in self.vocabs:
            self.vocabs["pretrained_word"] = utils.data.Vocab.fromkeys([], unknown)

    def load_embeddings(self, file, unknown="<UNK>", preprocess=str.lower):
        embeddings = utils.data.load_embeddings(file)
        dim = len(next(iter(embeddings.values())))
        embeddings[preprocess("<ROOT>")] = [0.0] * dim
        if unknown not in embeddings:
            embeddings[unknown] = [0.0] * dim
        self.vocabs["pretrained_word"] = utils.data.Vocab.fromkeys(embeddings.keys(), unknown)
        self.vocabs["pretrained_word"].preprocess = preprocess
        self._embeddings = list(embeddings.values())
        self._embed_file = file

    def transform(self, tokens):
        words, postags, heads, rels = zip(
            *[(token["form"], token["postag"], token["head"], token["deprel"]) for token in tokens]
        )
        word_ids = [self.vocabs["word"][s] for s in words]
        pre_ids = [self.vocabs["pretrained_word"][s] for s in words]
        postag_ids = [self.vocabs["postag"][s] for s in postags]
        rel_ids = [self.vocabs["rel"][s] for s in rels]
        return (word_ids, pre_ids, postag_ids, list(heads), rel_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.serialize_embeddings:
            state["_embeddings"] = None
        return state

    @property
    def pretrained_word_embeddings(self):
        if self._embeddings is None and self._embed_file is not None:
            v = self.vocabs["pretrained_word"]
            self.load_embeddings(self._embed_file, v.lookup(v.unknown_id), v.preprocess)
            assert len(self.vocabs["pretrained_word"]) == len(v)
        return self._embeddings


def create_dataloader(file, preprocessor: Preprocessor, device=None, **kwargs):
    dataset = ListDataset(map(preprocessor.transform, utils.conll.read_conll(file)))
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
