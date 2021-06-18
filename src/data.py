from collections import Counter

from utils.conll import read_conll
from utils.data import Vocab, load_embeddings


class Preprocessor:
    serialize_embeddings = False

    def __init__(self):
        self.vocabs = {}
        self._embeddings = None
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
        if "pretrained_word" not in self.vocabs:
            self.vocabs["pretrained_word"] = Vocab.fromkeys([], unknown)

    def load_embeddings(self, file, unknown="<UNK>", preprocess=str.lower):
        embeddings = load_embeddings(file)
        dim = len(next(iter(embeddings.values())))
        embeddings[preprocess("<ROOT>")] = [0.0] * dim
        if unknown not in embeddings:
            embeddings[unknown] = [0.0] * dim
        self.vocabs["pretrained_word"] = Vocab.fromkeys(embeddings.keys(), unknown)
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
