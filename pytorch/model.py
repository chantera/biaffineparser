from collections import defaultdict
import re

import numpy as np
from teras.base.event import Callback
from teras.dataset.loader import CorpusLoader
from teras.framework.pytorch.model import Biaffine, Embed, MLP
from teras.io.reader import ConllReader
import teras.logging as Log
from teras.preprocessing import text
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBiaffine(nn.Module):

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
                 label_mlp_dropout=0.33,
                 pad_id=0):
        super(DeepBiaffine, self).__init__()
        self._pad_id = pad_id
        blstm_cls = nn.GRU if use_gru else nn.LSTM
        self.embed = Embed(*embeddings, dropout=embeddings_dropout,
                           padding_idx=pad_id)
        embed_size = self.embed.size
        self.blstm = blstm_cls(
            num_layers=n_blstm_layers,
            input_size=embed_size,
            hidden_size=(lstm_hidden_size
                         if lstm_hidden_size is not None else embed_size),
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        layers = [MLP.Layer(lstm_hidden_size * 2, n_arc_mlp_units,
                            mlp_activation, arc_mlp_dropout)
                  for i in range(n_arc_mlp_layers)]
        self.mlp_arc_head = MLP(layers)
        layers = [MLP.Layer(lstm_hidden_size * 2, n_arc_mlp_units,
                            mlp_activation, arc_mlp_dropout)
                  for i in range(n_arc_mlp_layers)]
        self.mlp_arc_dep = MLP(layers)
        layers = [MLP.Layer(lstm_hidden_size * 2, n_label_mlp_units,
                            mlp_activation, label_mlp_dropout)
                  for i in range(n_label_mlp_layers)]
        self.mlp_label_head = MLP(layers)
        layers = [MLP.Layer(lstm_hidden_size * 2, n_label_mlp_units,
                            mlp_activation, label_mlp_dropout)
                  for i in range(n_label_mlp_layers)]
        self.mlp_label_dep = MLP(layers)
        self.arc_biaffine = \
            Biaffine(n_arc_mlp_units, n_arc_mlp_units, 1,
                     bias=(True, False, False))
        self.label_biaffine = \
            Biaffine(n_label_mlp_units, n_label_mlp_units, n_labels,
                     bias=(True, True, True))

    def forward(self, word_tokens, pos_tokens):
        lengths = np.array([len(tokens) for tokens in word_tokens])
        word_tokens = [np.pad(words, (0, lengths.max() - length),
                              mode="constant", constant_values=self._pad_id)
                       for words, length in zip(word_tokens, lengths)]
        pos_tokens = [np.pad(tags, (0, lengths.max() - length),
                             mode="constant", constant_values=self._pad_id)
                      for tags, length in zip(pos_tokens, lengths)]
        X = self.embed(word_tokens, pos_tokens)
        indices = np.argsort(-np.array(lengths)).astype('i')
        lengths = lengths[indices]
        X = torch.stack([X[idx] for idx in indices])
        X = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        R = self.blstm(X)[0]
        R = nn.utils.rnn.pad_packed_sequence(R, batch_first=True)[0]
        H_arc_head = self.mlp_arc_head(R)
        H_arc_dep = self.mlp_arc_dep(R)
        arc_logits = self.arc_biaffine(H_arc_dep, H_arc_head)
        arc_logits = torch.squeeze(arc_logits, dim=3)
        H_label_dep = self.mlp_label_dep(R)
        H_label_head = self.mlp_label_head(R)
        label_logits = self.label_biaffine(H_label_dep, H_label_head)
        return arc_logits, label_logits


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


def _model_var(model, x):
    p = next(model.parameters())
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


class BiaffineParser(object):

    def __init__(self, model, root_label=0):
        self.model = model
        self._ROOT_LABEL = root_label

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
        pred_arcs = torch.squeeze(
            torch.max(arc_logits, dim=1)[1], dim=1).data.cpu().numpy()
        label_logits = torch.transpose(label_logits, 1, 2)
        size = label_logits.size()
        output_logits = _model_var(
            self.model,
            torch.zeros(size[0], size[2], size[3]))
        for batch_index, (_logits, _arcs, _length) \
                in enumerate(zip(label_logits, pred_arcs, lengths)):
            for i in range(_length):
                output_logits[batch_index] = _logits[_arcs[i]]
        return output_logits

    def compute_loss(self, y, t):
        arc_logits, label_logits = y
        true_arcs, true_labels = t.T

        b, l1, l2 = arc_logits.size()
        true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, padding=-1, dtype=np.int64))
        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1)

        b, l1, d = label_logits.size()
        true_labels = _model_var(
            self.model,
            pad_sequence(true_labels, padding=-1, dtype=np.int64))
        label_loss = F.cross_entropy(
            label_logits.view(b * l1, d), true_labels.view(b * l1),
            ignore_index=-1)

        loss = arc_loss + label_loss
        return loss

    def compute_accuracy(self, y, t):
        arc_logits, label_logits = y
        true_arcs, true_labels = t.T

        b, l1, l2 = arc_logits.size()
        pred_arcs = arc_logits.data.max(2)[1].cpu()
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        correct = pred_arcs.eq(true_arcs).cpu().sum()
        arc_accuracy = (correct /
                        (b * l1 - np.sum(true_arcs.cpu().numpy() == -1)))

        b, l1, d = label_logits.size()
        pred_labels = label_logits.data.max(2)[1].cpu()
        true_labels = pad_sequence(true_labels, padding=-1, dtype=np.int64)
        correct = pred_labels.eq(true_labels).cpu().sum()
        label_accuracy = (correct /
                          (b * l1 - np.sum(true_labels.cpu().numpy() == -1)))

        accuracy = (arc_accuracy + label_accuracy) / 2
        return accuracy

    def parse(self, word_tokens=None, pos_tokens=None):
        if word_tokens is not None:
            self.forward(word_tokens, pos_tokens)
        ROOT = self._ROOT_LABEL
        arcs_batch, labels_batch = [], []
        arc_logits = self._arc_logits.data.cpu().numpy()
        label_logits = self._label_logits.data.cpu().numpy()

        for arc_logit, label_logit, length in \
                zip(arc_logits, np.transpose(label_logits, (0, 2, 1, 3)),
                    self._lengths):
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
    PUNCT_TAGS = ['``', "''", ':', ',', '.']

    def __init__(self, parser, pos_map=None, ignore_punct=True,
                 name='evaluator', **kwargs):
        super(Evaluator, self).__init__(name, **kwargs)
        self._parser = parser
        punct_ids = [pos_map[punct] if punct in pos_map else -2 for punct
                     in Evaluator.PUNCT_TAGS] if pos_map is not None else []
        self._PUNCTS = np.array(punct_ids)
        self._ignore_punct = ignore_punct
        self.reset()

    def reset(self):
        self.record = {'UAS': 0, 'LAS': 0, 'count': 0}

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

    def create_ignore_mask(self, words, postags, ignore_punct=True):
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
        word_tokens, pos_tokens = data['xs']
        arcs_batch, labels_batch = self._parser.parse()
        true_arcs, true_labels = data['ts'].T
        for i, (p_arcs, p_labels, t_arcs, t_labels) in enumerate(
                zip(arcs_batch, labels_batch, true_arcs, true_labels)):
            mask = self.create_ignore_mask(word_tokens[i], pos_tokens[i],
                                           self._ignore_punct)
            UAS, LAS, count = self.evaluate(p_arcs, p_labels,
                                            t_arcs, t_labels, mask)
            self.record['UAS'] += UAS
            self.record['LAS'] += LAS
            self.record['count'] += count

    def on_epoch_validate_end(self, data):
        self.report()
        self.reset()


class DataLoader(CorpusLoader):

    def __init__(self,
                 word_embed_size=100,
                 pos_embed_size=100,
                 word_embed_file=None,
                 word_preprocess=lambda x: re.sub(r'[0-9]', '0', x.lower()),
                 word_unknown="UNKNOWN"):
        super(DataLoader, self).__init__(reader=ConllReader())
        self.use_pretrained = word_embed_file is not None
        self.add_processor(
            'word', embed_file=word_embed_file, embed_size=word_embed_size,
            preprocess=word_preprocess, unknown=word_unknown, pad="<PAD>")
        self.add_processor(
            'pos', embed_file=None, embed_size=pos_embed_size,
            preprocess=lambda x: x.lower(), pad="<PAD>")
        self.label_map = text.Vocab()
        self._sentences = {}

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
        sentence_id = ':'.join(str(word_id) for word_id in sample[0])
        self._sentences[sentence_id] = words
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

    def get_sentence(self, word_ids, default=None):
        sentence_id = ':'.join(str(word_id) for word_id in word_ids)
        return self._sentences.get(sentence_id, default)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sentences'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
