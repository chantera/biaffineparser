from collections import defaultdict
import re

import numpy as np
from teras.base.event import Callback
from teras.dataset.loader import CorpusLoader
from teras.io.reader import ConllReader
import teras.logging as Log
from teras.preprocessing import text


def mst(scores):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692  # NOQA
    """
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = 0
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        nonlocal _index
        _indices[v] = _index
        _lowlinks[v] = _index
        _index += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not(w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


class Evaluator(Callback):
    PUNCT_TAGS = ['``', "''", ':', ',', '.']

    def __init__(self, parser, pos_map=None, ignore_punct=True,
                 name='evaluator', **kwargs):
        super(Evaluator, self).__init__(name, **kwargs)
        self._parser = parser
        punct_ids = [pos_map[punct] if punct in pos_map else -2 for punct
                     in Evaluator.PUNCT_TAGS] if pos_map is not None else []
        self._PUNCTS = np.array(punct_ids)
        self._ignore_punct = ignore_punct
        self.reset(init_history=True)

    def reset(self, init_history=False):
        if init_history:
            self._history = []
        self.record = {'UAS': 0, 'LAS': 0, 'count': 0}

    def get_history(self):
        return self._history

    def on_train_begin(self, data):
        self.reset(init_history=True)

    def evaluate(self, pred_arcs, pred_labels,
                 true_arcs, true_labels, ignore_mask=None):
        if ignore_mask is None:
            ignore_mask = np.zeros(len(true_arcs), np.int32)
            ignore_mask[0] = 1
        mask = 1 - ignore_mask
        count = np.sum(mask)
        match_arcs = (pred_arcs == true_arcs) * mask
        UAS = np.sum(match_arcs)
        match_labels = pred_labels == true_labels
        LAS = np.sum((match_arcs * match_labels))
        return UAS, LAS, count

    def create_ignore_mask(self, postags, ignore_punct=True):
        if ignore_punct:
            mask = np.isin(postags, self._PUNCTS).astype(np.int32)
        else:
            mask = np.zeros(len(postags), np.int32)
        mask[0] = 1
        return mask

    def report(self):
        Log.i("[evaluation] UAS: {:.8f}, LAS: {:.8f}"
              .format(self.record['UAS'] / self.record['count'] * 100,
                      self.record['LAS'] / self.record['count'] * 100))

    def on_batch_end(self, data):
        if data['train']:
            return
        pos_tokens = data['xs'][2]
        arcs_batch, labels_batch = self._parser.parse()
        true_arcs, true_labels = data['ts'].T
        for i, (p_arcs, p_labels, t_arcs, t_labels) in enumerate(
                zip(arcs_batch, labels_batch, true_arcs, true_labels)):
            mask = self.create_ignore_mask(pos_tokens[i], self._ignore_punct)
            UAS, LAS, count = self.evaluate(p_arcs, p_labels,
                                            t_arcs, t_labels, mask)
            self.record['UAS'] += UAS
            self.record['LAS'] += LAS
            self.record['count'] += count

    def on_epoch_validate_end(self, data):
        self.report()
        self._history.append({
            'UAS': self.record['UAS'] / self.record['count'] * 100,
            'LAS': self.record['LAS'] / self.record['count'] * 100,
            'count': self.record['count']
        })
        self.reset()


class DataLoader(CorpusLoader):

    def __init__(self,
                 word_embed_size=100,
                 pos_embed_size=100,
                 word_embed_file=None,
                 word_preprocess=lambda x: re.sub(r'[0-9]', '0', x.lower()),
                 word_unknown="UNKNOWN",
                 embed_dtype='float32'):
        super(DataLoader, self).__init__(reader=ConllReader())
        if word_embed_file is not None:
            self.use_pretrained = True
            self.add_processor(
                'word_pretrained', embed_file=word_embed_file,
                embed_dtype=embed_dtype,
                preprocess=word_preprocess, unknown=word_unknown)
            word_min_frequency = 2
        else:
            self.add_processor(
                'word_pretrained', embed_size=word_embed_size,
                embed_dtype=embed_dtype,
                preprocess=word_preprocess, unknown=word_unknown,
                initializer=np.zeros)
            self.use_pretrained = False
            word_min_frequency = 1
        self.add_processor(
            'word', embed_size=word_embed_size,
            embed_dtype=embed_dtype,
            preprocess=word_preprocess, unknown=word_unknown,
            min_frequency=word_min_frequency)
        self.add_processor(
            'pos', embed_size=pos_embed_size,
            embed_dtype=embed_dtype,
            preprocess=lambda x: x.lower())
        self.label_map = text.Vocab()
        self._sentences = {}

    def map(self, item):
        # item -> (pretrained_words, words, postags, (heads, labels))
        words, postags, heads, labels = \
            zip(*[(token['form'],
                   token['postag'],
                   token['head'],
                   self.label_map.add(token['deprel']))
                  for token in item])
        sample = (self.get_processor('word_pretrained').transform_one(words),
                  self._word_transform_one(words),
                  self.get_processor('pos').fit_transform_one(postags),
                  (np.array(heads, dtype=np.int32),
                   np.array(labels, dtype=np.int32)))
        self._sentences[hash(tuple(sample[0]))] = words
        return sample

    def load(self, file, train=False, size=None):
        if train:
            # assign an index if the given word is not in vocabulary
            self._word_transform_one = \
                self.get_processor('word').fit_transform_one
        else:
            # return the unknown word index if the word is not in vocabulary
            self._word_transform_one = \
                self.get_processor('word').transform_one
        return super(DataLoader, self).load(file, train, size)

    def get_sentence(self, word_ids, default=None):
        return self._sentences.get(hash(tuple(word_ids)), default)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sentences'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
