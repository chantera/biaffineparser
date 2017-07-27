from chainer import Chain, cuda
import chainer.functions as F
import numpy as np
from teras.framework.chainer.model import Biaffine, BiGRU, BiLSTM, Embed, MLP

from utils import mst


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
        super(DeepBiaffine, self).__init__()
        blstm_cls = BiGRU if use_gru else BiLSTM
        with self.init_scope():
            self.embed = Embed(*embeddings, dropout=embeddings_dropout)
            embed_size = self.embed.size
            self.blstm = blstm_cls(
                n_layers=n_blstm_layers,
                in_size=embed_size,
                out_size=(lstm_hidden_size
                          if lstm_hidden_size is not None else embed_size),
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
        if not self.model._cpu:
            true_arcs.to_gpu()
        arc_loss = F.softmax_cross_entropy(
            F.reshape(arc_logits, (b * l1, l2)),
            F.reshape(true_arcs, (b * l1,)),
            ignore_label=-1)

        b, l1, d = label_logits.shape
        true_labels = F.pad_sequence(true_labels, padding=-1)
        if not self.model._cpu:
            true_labels.to_gpu()
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
        if not self.model._cpu:
            true_arcs.to_gpu()
        arc_accuracy = F.accuracy(
            F.reshape(arc_logits, (b * l1, l2)),
            F.reshape(true_arcs, (b * l1,)),
            ignore_label=-1)

        b, l1, d = label_logits.shape
        true_labels = F.pad_sequence(true_labels, padding=-1)
        if not self.model._cpu:
            true_labels.to_gpu()
        label_accuracy = F.accuracy(
            F.reshape(label_logits, (b * l1, d)),
            F.reshape(true_labels, (b * l1,)),
            ignore_label=-1)

        accuracy = (arc_accuracy + label_accuracy) / 2
        return accuracy

    def parse(self, word_tokens=None, pos_tokens=None):
        if word_tokens is not None:
            self.forward(word_tokens, pos_tokens)
        ROOT = self._ROOT_LABEL
        arcs_batch, labels_batch = [], []
        arc_logits = cuda.to_cpu(self._arc_logits.data)
        label_logits = cuda.to_cpu(self._label_logits.data)

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
