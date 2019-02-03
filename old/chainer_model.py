from chainer import Chain, cuda, initializer, initializers
import chainer.functions as F
import numpy as np
import teras.framework.chainer.functions as teras_F
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
            orthonormal_initializer = Orthonormal()
            zero_initializer = initializers.Zero()
            self.embed = Embed(*embeddings, dropout=embeddings_dropout)
            embed_size = self.embed.size - self.embed[0].W.shape[1]
            self.blstm = blstm_cls(
                n_layers=n_blstm_layers,
                in_size=embed_size,
                out_size=(lstm_hidden_size
                          if lstm_hidden_size is not None else embed_size),
                dropout=lstm_dropout,
                initialW=orthonormal_initializer
            )
            layers = [MLP.Layer(None, n_arc_mlp_units,
                                mlp_activation, arc_mlp_dropout,
                                initialW=orthonormal_initializer)
                      for i in range(n_arc_mlp_layers)]
            self.mlp_arc_head = MLP(layers)
            layers = [MLP.Layer(None, n_arc_mlp_units,
                                mlp_activation, arc_mlp_dropout,
                                initialW=orthonormal_initializer)
                      for i in range(n_arc_mlp_layers)]
            self.mlp_arc_dep = MLP(layers)
            layers = [MLP.Layer(None, n_label_mlp_units,
                                mlp_activation, label_mlp_dropout,
                                initialW=orthonormal_initializer)
                      for i in range(n_label_mlp_layers)]
            self.mlp_label_head = MLP(layers)
            layers = [MLP.Layer(None, n_label_mlp_units,
                                mlp_activation, label_mlp_dropout,
                                initialW=orthonormal_initializer)
                      for i in range(n_label_mlp_layers)]
            self.mlp_label_dep = MLP(layers)
            self.arc_biaffine = \
                Biaffine(n_arc_mlp_units, n_arc_mlp_units, 1,
                         nobias=(False, True, True),
                         initialW=zero_initializer)
            self.label_biaffine = \
                Biaffine(n_label_mlp_units, n_label_mlp_units, n_labels,
                         nobias=(False, False, False),
                         initialW=zero_initializer)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, pretrained_word_tokens, word_tokens, pos_tokens):
        X = []
        batch = len(word_tokens)
        for i in range(batch):
            xs_words_pretrained = \
                self.embed[0](self.xp.array(pretrained_word_tokens[i]))
            xs_words = self.embed[1](self.xp.array(word_tokens[i]))
            xs_words += xs_words_pretrained
            xs_tags = self.embed[2](self.xp.array(pos_tokens[i]))
            xs = F.concat([
                teras_F.dropout(xs_words, self.embed._dropout_ratio),
                teras_F.dropout(xs_tags, self.embed._dropout_ratio)])
            X.append(xs)
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


class Orthonormal(initializer.Initializer):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/linalg.py#L35  # NOQA
    """
    _logger = None

    def __init__(self, dtype=None):
        if self._logger is None:
            import logging
            Orthonormal._logger = logging.getLogger()
        super(Orthonormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        output_size, input_size = array.shape
        I = xp.eye(output_size)
        lr = .1
        eps = .05 / (output_size + input_size)
        success = False
        tries = 0
        while not success and tries < 10:
            Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
            for i in range(100):
                QTQmI = Q.T.dot(Q) - I
                loss = xp.sum(QTQmI ** 2 / 2)
                Q2 = Q ** 2
                Q -= lr * Q.dot(QTQmI) / \
                    (xp.abs(Q2 + Q2.sum(axis=0, keepdims=True)
                            + Q2.sum(axis=1, keepdims=True) - 1) + eps)
                if xp.max(Q) > 1e6 or loss > 1e6 or not xp.isfinite(loss):
                    tries += 1
                    lr /= 2
                    break
            success = True
        if success:
            self._logger.trace('Orthogonal pretrainer loss: %.2e' % loss)
        else:
            self._logger.trace('Orthogonal pretrainer failed, '
                               'using non-orthogonal random matrix')
        Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
        array[...] = Q.T


class BiaffineParser(object):

    def __init__(self, model, root_label=0):
        self.model = model
        self._ROOT_LABEL = root_label

    def forward(self, pretrained_word_tokens, word_tokens, pos_tokens):
        lengths = [len(tokens) for tokens in word_tokens]
        arc_logits, label_logits = \
            self.model.forward(pretrained_word_tokens, word_tokens, pos_tokens)
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

    def parse(self, pretrained_word_tokens=None,
              word_tokens=None, pos_tokens=None):
        if word_tokens is not None:
            self.forward(pretrained_word_tokens, word_tokens, pos_tokens)
        ROOT = self._ROOT_LABEL
        arcs_batch, labels_batch = [], []
        arc_logits = cuda.to_cpu(self._arc_logits.data)
        label_logits = cuda.to_cpu(self._label_logits.data)

        for arc_logit, label_logit, length in \
                zip(arc_logits, np.transpose(label_logits, (0, 2, 1, 3)),
                    self._lengths):
            arc_probs = softmax2d(arc_logit[:length, :length])
            arcs = mst(arc_probs)
            label_probs = softmax2d(label_logit[np.arange(length), arcs])
            labels = np.argmax(label_probs, axis=1)
            labels[0] = ROOT
            tokens = np.arange(1, length)
            roots = np.where(labels[tokens] == ROOT)[0] + 1
            if len(roots) < 1:
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[root_arc] = ROOT
            elif len(roots) > 1:
                label_probs[roots, ROOT] = 0
                new_labels = \
                    np.argmax(label_probs[roots], axis=1)
                root_arc = np.where(arcs[tokens] == 0)[0] + 1
                labels[roots] = new_labels
                labels[root_arc] = ROOT
            arcs_batch.append(arcs)
            labels_batch.append(labels)

        return arcs_batch, labels_batch


def softmax2d(x):
    y = x - np.max(x, axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y
