from itertools import accumulate

import chainer
import chainer.functions as F
import numpy as np


class EmbedID(chainer.link.Link):
    """Same as `chainer.links.EmbedID` except for fixing pretrained weight."""

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None,
                 fix_weight=False):
        super().__init__()
        self.ignore_label = ignore_label
        self.fix_weight = fix_weight

        with self.init_scope():
            if initialW is None:
                initialW = chainer.initializers.normal.Normal(1.0)
            self.W = chainer.Parameter(initialW, (in_size, out_size))
            if fix_weight:
                self.W._requires_grad = self.W._node._requires_grad = False

    def forward(self, x):
        return embed_id(x, self.W, self.ignore_label, self.fix_weight)


def embed_id(x, W, ignore_label=None, fix_weight=False):
    return _EmbedIDFunction(ignore_label, fix_weight).apply((x, W))[0]


class _EmbedIDFunction(F.connection.embed_id.EmbedIDFunction):

    def __init__(self, ignore_label=None, fix_weight=False):
        super().__init__(ignore_label)
        self.fix_weight = fix_weight

    def backward(self, indexes, grad_outputs):
        if self.fix_weight:
            return None, None
        return super().backward(indexes, grad_outputs)


class EmbedList(chainer.link.ChainList):

    def __init__(self, embeddings, dropout=0.0, merge=True):
        assert all(isinstance(embed, (chainer.links.EmbedID, EmbedID))
                   for embed in embeddings)
        super().__init__(*embeddings)
        if isinstance(dropout, (list, tuple)) \
                and len(dropout) != len(embeddings):
            raise ValueError(
                "dropout ratio must be specified for all embeddings")
        self.dropout = dropout
        self.merge = merge

    def forward(self, *xs, merged=True):
        if isinstance(self.dropout, (list, tuple)):
            dropout_each, dropout_all = self.dropout, 0.0
        else:
            dropout_each, dropout_all = [0.0] * len(self), self.dropout
        boundaries = list(accumulate(len(x) for x in xs[0][:-1]))
        xp = chainer.cuda.get_array_module(xs[0][0])
        ys_flatten = [_apply_dropout(
            embed(xp.concatenate(xs_each, axis=0)), dropout)
            for embed, xs_each, dropout in zip(self, xs, dropout_each)]
        if self.merge:
            ys = F.split_axis(_apply_dropout(
                F.concat(ys_flatten), dropout_all), boundaries, axis=0)
        else:
            ys = [F.split_axis(ys_flatten_each, boundaries, axis=0)
                  for ys_flatten_each in ys_flatten]
        return ys


class MLP(chainer.link.ChainList):

    def __init__(self, layers):
        assert all(isinstance(layer, MLP.Layer) for layer in layers)
        super().__init__(*layers)

    def forward(self, x, n_batch_axes=1):
        for layer in self:
            x = layer(x, n_batch_axes)
        return x

    class Layer(chainer.links.Linear):

        def __init__(self, in_size, out_size=None,
                     activation=None, dropout=0.0,
                     nobias=False, initialW=None, initial_bias=None):
            super().__init__(in_size, out_size, nobias, initialW, initial_bias)
            if activation is not None and not callable(activation):
                raise TypeError("activation must be callable: type={}"
                                .format(type(activation)))
            self.activate = activation
            self.dropout = dropout

        def forward(self, x, n_batch_axes=1):
            h = super().forward(x, n_batch_axes)
            if self.activate is not None:
                h = self.activate(h)
            return _apply_dropout(h, self.dropout)


def _apply_dropout(x, ratio=.5, **kwargs):
    """Disable dropout when ratio == 0.0."""
    if chainer.configuration.config.train and ratio > 0.0:
        return F.dropout(x, ratio, **kwargs)
    return chainer.as_variable(x)


dropout = _apply_dropout


class Biaffine(chainer.link.Link):

    def __init__(self, left_size, right_size, out_size,
                 nobias=(False, False, False),
                 initialW=None, initial_bias=None):
        super().__init__()
        self.in_sizes = (left_size, right_size)
        self.out_size = out_size
        self.nobias = nobias

        with self.init_scope():
            shape = (left_size + int(not(self.nobias[0])),
                     right_size + int(not(self.nobias[1])),
                     out_size)
            if isinstance(initialW, (np.ndarray, chainer.cuda.ndarray)):
                assert initialW.shape == shape
            self.W = chainer.Parameter(
                chainer.initializers._get_initializer(initialW), shape)

            if not self.nobias[2]:
                if initial_bias is None:
                    initial_bias = 0
                self.b = chainer.Parameter(initial_bias, (out_size,))

    def forward(self, x1, x2):
        xp = self.xp
        out_size = self.out_size
        batch_size, n1, d1 = x1.shape
        if not self.nobias[0]:
            x1 = F.concat(
                (x1, xp.ones((batch_size, n1, 1), xp.float32)), axis=2)
            d1 += 1
        n2, d2 = x2.shape[1:]
        if not self.nobias[1]:
            x2 = F.concat(
                (x2, xp.ones((batch_size, n2, 1), xp.float32)), axis=2)
            d2 += 1
        # (B * n1, d1) @ (d1, O * d2) => (B * n1, O * d2)
        x1W = F.matmul(
            F.reshape(x1, (batch_size * n1, d1)),
            F.reshape(F.transpose(self.W, (0, 2, 1)), (d1, out_size * d2)))
        # (B, n1 * O, d2) @ (B, d2, n2) => (B, n1 * O, n2)
        x1Wx2 = F.matmul(
            F.reshape(x1W, (batch_size, n1 * out_size, d2)),
            x2, transb=True)
        # => (B, n1, n2, O)
        y = F.transpose(F.reshape(x1Wx2, (batch_size, n1, out_size, n2)),
                        (0, 1, 3, 2))
        assert y.shape == (batch_size, n1, n2, out_size)
        if not self.nobias[2]:
            y += F.broadcast_to(self.b, y.shape)
        return y
