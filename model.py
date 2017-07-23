from collections import defaultdict
import re

from chainer import Chain, cuda
import chainer.functions as F
import numpy as np
from teras.base.event import Callback
from teras.dataset.loader import CorpusLoader
from teras.framework.chainer.model import Biaffine, BiGRU, BiLSTM, Embed, MLP
from teras.io.reader import ConllReader
import teras.logging as Log
from teras.preprocessing import text


class DeepBiaffine(Chain):

    def __init__(self,
                 embeddings,
                 n_labels,
                 n_blstm_layers=3,
                 lstm_hidden_size=400,
                 use_gru=False,
                 n_arc_mlp_layers=1,
                 n_arc_mlp_units=500,
                 n_label_mlp_layers=1,
                 n_label_mlp_units=100,
                 mlp_activation=F.leaky_relu,
                 embeddings_dropout=0.33,
                 lstm_dropout=0.33,
                 arc_mlp_dropout=0.33,
                 label_mlp_dropout=0.33):
        embed_size = sum(embed.shape[1] for embed in embeddings)
        if lstm_hidden_size is None:
            lstm_hidden_size = embed_size
        super(DeepBiaffine, self).__init__()
        blstm_cls = BiGRU if use_gru else BiLSTM
        with self.init_scope():
            self.embed = Embed(*embeddings, dropout=embeddings_dropout)
            self.blstm = blstm_cls(
                n_layers=n_blstm_layers,
                in_size=embed_size,
                out_size=lstm_hidden_size,
                dropout=lstm_dropout
            )
            layers = [MLP.Layer(None, n_arc_mlp_units,
                                mlp_activation, arc_mlp_dropout)
                      for i in range(n_arc_mlp_layers)]
            self.mlp_arc_head = MLP(layers)
            layers = [MLP.Layer(None, n_arc_mlp_units,
                                mlp_activation, arc_mlp_dropout)
                      for i in range(n_arc_mlp_layers)]
            self.mlp_arc_dep = MLP(layers)
            layers = [MLP.Layer(None, n_label_mlp_units,
                                mlp_activation, label_mlp_dropout)
                      for i in range(n_label_mlp_layers)]
            self.mlp_label_head = MLP(layers)
            layers = [MLP.Layer(None, n_label_mlp_units,
                                mlp_activation, label_mlp_dropout)
                      for i in range(n_label_mlp_layers)]
            self.mlp_label_dep = MLP(layers)
            self.arc_biaffine = \
                Biaffine(n_arc_mlp_units, n_arc_mlp_units, 1,
                         nobias=(False, True, True))
            self.label_biaffine = \
                Biaffine(n_label_mlp_units, n_label_mlp_units, n_labels,
                         nobias=(False, False, False))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, word_tokens, pos_tokens):
        X = self.embed(word_tokens, pos_tokens)
        R = self.blstm(X)
        R = F.pad_sequence(R)
        H_arc_dep = self.mlp_arc_dep(R)
        H_arc_head = self.mlp_arc_head(R)
        arc_logits = self.arc_biaffine(H_arc_dep, H_arc_head)
        arc_logits = F.squeeze(arc_logits, axis=3)
        H_label_dep = self.mlp_label_dep(R)
        H_label_head = self.mlp_label_head(R)
        label_logits = self.label_biaffine(H_label_dep, H_label_head)
        return arc_logits, label_logits


class BiaffineParser(object):

    def __init__(self, model, label_map):
        self.model = model
        self.label_map = label_map

    def forward(self, word_tokens, pos_tokens):
        lengths = [len(tokens) for tokens in word_tokens]
        arc_logits, label_logits = self.model.forward(word_tokens, pos_tokens)
        # cache
        self._lengths = lengths
        self._arc_logits = arc_logits
        self._label_logits = label_logits

        label_logits = \
            self.extract_best_label_logits(arc_logits, label_logits, lengths)
        return arc_logits, label_logits

    def extract_best_label_logits(self, arc_logits, label_logits, lengths):
        pred_arcs = self.model.xp.argmax(arc_logits.data, axis=1)
        label_logits = F.transpose(label_logits, (0, 2, 1, 3))
        label_logits = [_logits[np.arange(_length), _arcs[:_length]]
                        for _logits, _arcs, _length
                        in zip(label_logits, pred_arcs, lengths)]
        label_logits = F.pad_sequence(label_logits)
        return label_logits

    def compute_loss(self, y, t):
        arc_logits, label_logits = y
        true_arcs, true_labels = t.T

        b, l1, l2 = arc_logits.shape
        true_arcs = F.pad_sequence(true_arcs, padding=-1)
        arc_loss = F.softmax_cross_entropy(
            F.reshape(arc_logits, (b * l1, l2)),
            F.reshape(true_arcs, (b * l1,)),
            ignore_label=-1)

        b, l1, d = label_logits.shape
        true_labels = F.pad_sequence(true_labels, padding=-1)
        label_loss = F.softmax_cross_entropy(
            F.reshape(label_logits, (b * l1, d)),
            F.reshape(true_labels, (b * l1,)),
            ignore_label=-1)

        loss = arc_loss + label_loss
        return loss

    def compute_accuracy(self, y, t):
        arc_logits, label_logits = y
        true_arcs, true_labels = t.T

        b, l1, l2 = arc_logits.shape
        true_arcs = F.pad_sequence(true_arcs, padding=-1)
        arc_accuracy = F.accuracy(
            F.reshape(arc_logits, (b * l1, l2)),
            F.reshape(true_arcs, (b * l1,)),
            ignore_label=-1)

        b, l1, d = label_logits.shape
        true_labels = F.pad_sequence(true_labels, padding=-1)
        label_accuracy = F.accuracy(
            F.reshape(label_logits, (b * l1, d)),
            F.reshape(true_labels, (b * l1,)),
            ignore_label=-1)

        accuracy = (arc_accuracy + label_accuracy) / 2
        return accuracy

    def parse(self, word_tokens=None, pos_tokens=None):
        if word_tokens is not None:
            self.forward(word_tokens, pos_tokens)
        ROOT = self.label_map['root']

        arcs_batch, labels_batch = [], []
        arc_logits = cuda.to_cpu(self._arc_logits.data)
        label_logits = cuda.to_cpu(self._label_logits.data)

        for arc_logit, label_logit, length in \
                zip(arc_logits, label_logits, self._lengths):
            arcs = mst(arc_logit[:length, :length])
            label_scores = label_logit[np.arange(length), arcs]
            labels = np.argmax(label_scores, axis=1)
            labels[0] = ROOT
            tokens = np.arange(1, length)
            roots = np.where(labels[tokens] == ROOT)[0] + 1
            if len(roots) < 1:
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[root_arc] = ROOT
            elif len(roots) > 1:
                label_scores[roots, ROOT] = 0
                new_labels = \
                    np.argmax(label_scores[roots], axis=1)
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[roots] = new_labels
                labels[root_arc] = ROOT
            arcs_batch.append(arcs)
            labels_batch.append(labels)

        return arcs_batch, labels_batch


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

    def __init__(self, parser, name='evaluator', **kwargs):
        super(Evaluator, self).__init__(name, **kwargs)
        self._parser = parser
        self.reset()

    def reset(self):
        self.record = {'UAS': 0, 'LAS': 0, 'count': 0}

    def evaluate(self, pred_arcs, pred_labels, true_arcs, true_labels):
        count = len(true_arcs) - 1
        match_arcs = pred_arcs[1:] == true_arcs[1:]
        UAS = np.sum(match_arcs)
        match_labels = pred_labels[1:] == true_labels[1:]
        LAS = np.sum((match_arcs * match_labels))
        return UAS, LAS, count

    def report(self):
        Log.i("[evaluation] UAS: {:.8f}, LAS: {:.8f}"
              .format(self.record['UAS'] / self.record['count'] * 100,
                      self.record['LAS'] / self.record['count'] * 100))
        self.reset()

    def on_batch_end(self, data):
        if data['train']:
            return
        arcs_batch, labels_batch = self._parser.parse()
        true_arcs, true_labels = data['ts'].T
        for p_arcs, p_labels, t_arcs, t_labels in \
                zip(arcs_batch, labels_batch, true_arcs, true_labels):
            UAS, LAS, count = self.evaluate(p_arcs, p_labels, t_arcs, t_labels)
            self.record['UAS'] += UAS
            self.record['LAS'] += LAS
            self.record['count'] += count

    def on_epoch_validate_end(self, data):
        self.report()


class DataLoader(CorpusLoader):

    def __init__(self,
                 word_embed_size=100,
                 pos_embed_size=50,
                 word_embed_file=None,
                 word_preprocess=lambda x: re.sub(r'[0-9]', '0', x.lower()),
                 word_unknown="UNKNOWN"):
        super(DataLoader, self).__init__(reader=ConllReader())
        self.use_pretrained = word_embed_size is not None
        self.add_processor(
            'word', embed_file=word_embed_file, embed_size=word_embed_size,
            preprocess=word_preprocess, unknown=word_unknown)
        self.add_processor(
            'pos', embed_file=None, embed_size=pos_embed_size,
            preprocess=lambda x: x.lower())
        self.label_map = text.Vocab()

    def map(self, item):
        # item -> (words, postags, (heads, labels))
        words = []
        postags = []
        heads = []
        labels = []
        for token in item:
            words.append(token['form'])
            postags.append(token['postag'])
            heads.append(token['head'])
            labels.append(self.label_map.add(token['deprel']))
        sample = (self._word_transform_one(words),
                  self.get_processor('pos').fit_transform_one(postags),
                  (np.array(heads, dtype=np.int32),
                   np.array(labels, dtype=np.int32)))
        return sample

    def load(self, file, train=False, size=None):
        if train and not self.use_pretrained:
            # assign an index if the given word is not in vocabulary
            word_transform_one = self.get_processor('word').fit_transform_one
        else:
            # return the unknown word index if the word is not in vocabulary
            word_transform_one = self.get_processor('word').transform_one
        self._word_transform_one = word_transform_one
        return super(DataLoader, self).load(file, train, size)
