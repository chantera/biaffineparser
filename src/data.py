import itertools
import os
from collections import Counter

from utils.data import Vocab, load_embeddings


class Preprocessor:
    serialize_embeddings = False

    def __init__(self):
        self.vocabs = {}
        self._pretrained_embeddings = None
        self._embed_file = None

    def build_vocab(self, file, unknown="<UNK>", preprocess=str.lower, min_frequency=2):
        word_set, postag_set, rel_set = set(), set(), set()
        word_counter = Counter()
        for tokens in read_conll(file):
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

        self.vocabs["word"] = Vocab.fromkeys(word_set, unknown)
        self.vocabs["word"].preprocess = preprocess
        self.vocabs["postag"] = Vocab.fromkeys(postag_set, unknown)
        self.vocabs["rel"] = Vocab.fromkeys(rel_set)
        if "pretrained" not in self.vocabs:
            self.vocabs["pretrained"] = Vocab.fromkeys([], unknown)

    def load_embeddings(self, file, unknown="<UNK>", preprocess=str.lower):
        embeddings = load_embeddings(os.path.expanduser(file))
        dim = len(next(iter(embeddings.values())))
        embeddings[preprocess("<ROOT>")] = [0.0] * dim
        if unknown not in embeddings:
            embeddings[unknown] = [0.0] * dim
        self.vocabs["pretrained"] = Vocab.fromkeys(embeddings.keys(), unknown)
        self.vocabs["pretrained"].preprocess = preprocess
        self._pretrained_embeddings = list(embeddings.values())
        self._embed_file = file

    def transform(self, tokens):
        words, postags, heads, rels = zip(
            *[(token["form"], token["postag"], token["head"], token["deprel"]) for token in tokens]
        )
        word_ids = [self.vocabs["word"][s] for s in words]
        pre_ids = [self.vocabs["pretrained"][s] for s in words]
        postag_ids = [self.vocabs["postag"][s] for s in postags]
        rel_ids = [self.vocabs["rel"][s] for s in rels]
        return (word_ids, pre_ids, postag_ids, (heads, rel_ids))

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.serialize_embeddings:
            state["_pretrained_embeddings"] = None
        return state

    @property
    def pretrained_embeddings(self):
        if self._pretrained_embeddings is None and self._embed_file is not None:
            v = self.vocabs["pretrained"]
            self.load_embeddings(self._embed_file, v.lookup(v.unknown_id), v.preprocess)
            assert len(self.vocabs["pretrained"]) == len(v)
        return self._pretrained_embeddings


class Loader:
    def __init__(self, cache_dir=None):
        self.preprocessor = Preprocessor()

    def load(self, file, limit=None):
        examples = read_conll(file)
        if limit:
            itertools.islice(examples, limit)
        return list(map(self.preprocessor.transform, examples))


def read_conll(file):
    with open(os.path.expanduser(file)) as f:
        yield from parse_conll(f)


def parse_conll(lines):
    def _create_root():
        token = {
            "id": 0,
            "form": "<ROOT>",
            "lemma": "<ROOT>",
            "cpostag": "ROOT",
            "postag": "ROOT",
            "feats": "_",
            "head": 0,
            "deprel": "root",
            "phead": "_",
            "pdeprel": "_",
        }
        return token

    tokens = [_create_root()]
    for line in lines:
        line = line.strip()
        if not line:
            if len(tokens) > 1:
                yield tokens
                tokens = [_create_root()]
        elif line.startswith("#"):
            continue
        else:
            cols = line.split("\t")
            token = {
                "id": int(cols[0]),
                "form": cols[1],
                "lemma": cols[2],
                "cpostag": cols[3],
                "postag": cols[4],
                "feats": cols[5],
                "head": int(cols[6]),
                "deprel": cols[7],
                "phead": cols[8],
                "pdeprel": cols[9],
            }
            tokens.append(token)
    if len(tokens) > 1:
        yield tokens
