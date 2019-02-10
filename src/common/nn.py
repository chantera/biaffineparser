from itertools import accumulate

import chainer
import chainer.functions as F
import numpy as np


def _apply_dropout(x, ratio=.5, **kwargs):
    """Disable dropout when ratio == 0.0."""
    if chainer.configuration.config.train and ratio > 0.0:
        return F.dropout(x, ratio, **kwargs)
    out, mask = chainer.as_variable(x), None
    if kwargs.get('return_mask', False):
        return out, mask
    return out


dropout = _apply_dropout


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
            embed(self.xp.array(xp.concatenate(xs_each, axis=0))), dropout)
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


def _n_step_rnn_impl(
        f, n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        recurrent_dropout_ratio, use_variational_dropout):
    direction = 2 if use_bi_direction else 1
    hx = F.separate(hx)
    use_cell = cx is not None
    if use_cell:
        cx = F.separate(cx)
    else:
        cx = [None] * len(hx)

    xs_next = xs
    hy = []
    cy = []
    # NOTE(chantera):
    # Unlike Chainer, dropout is applied to inputs of the first layer
    # when using variational dropout.
    for layer in range(n_layers):
        # Forward RNN
        idx = direction * layer
        h_mask = None
        if use_variational_dropout:
            x_mask = _apply_dropout(
                xs_next[0].data, recurrent_dropout_ratio, return_mask=True)[1]
            h_mask = _apply_dropout(
                hx[idx].data, recurrent_dropout_ratio, return_mask=True)[1]
            xs = _dropout_sequence(xs_next, dropout_ratio, x_mask)
        elif layer == 0:
            xs = xs_next
        else:
            xs = _dropout_sequence(xs_next, dropout_ratio)
        h, c, h_forward = _one_directional_loop(
            f, xs, hx[idx], cx[idx], ws[idx], bs[idx],
            lambda h: _apply_dropout(h, recurrent_dropout_ratio, mask=h_mask))
        hy.append(h)
        cy.append(c)

        if use_bi_direction:
            # Backward RNN
            idx = direction * layer + 1
            h_mask = None
            if use_variational_dropout:
                x_mask = _apply_dropout(
                    xs_next[0].data, recurrent_dropout_ratio,
                    return_mask=True)[1]
                h_mask = _apply_dropout(
                    hx[idx].data, recurrent_dropout_ratio, return_mask=True)[1]
                xs = _dropout_sequence(xs_next, dropout_ratio, x_mask)
            elif layer == 0:
                xs = xs_next
            else:
                xs = _dropout_sequence(xs_next, dropout_ratio)
            h, c, h_backward = _one_directional_loop(
                f, reversed(xs), hx[idx], cx[idx], ws[idx], bs[idx],
                lambda h: _apply_dropout(
                    h, recurrent_dropout_ratio, mask=h_mask))
            h_backward.reverse()
            # Concat
            xs_next = [F.concat([hfi, hbi], axis=1)
                       for hfi, hbi in zip(h_forward, h_backward)]
            hy.append(h)
            cy.append(c)
        else:
            # Uni-directional RNN
            xs_next = h_forward

    ys = xs_next
    hy = F.stack(hy)
    if use_cell:
        cy = F.stack(cy)
    else:
        cy = None
    return hy, cy, tuple(ys)


def _one_directional_loop(f, xs, h, c, w, b, h_dropout):
    h_list = []
    for t, x in enumerate(xs):
        h = h_dropout(h)
        batch = len(x)
        need_split = len(h) > batch
        if need_split:
            h, h_rest = F.split_axis(h, [batch], axis=0)
            if c is not None:
                c, c_rest = F.split_axis(c, [batch], axis=0)

        h, c = f(x, h, c, w, b)
        h_list.append(h)

        if need_split:
            h = F.concat([h, h_rest], axis=0)
            if c is not None:
                c = F.concat([c, c_rest], axis=0)
    return h, c, h_list


def _dropout_sequence(xs, dropout_ratio, mask=None):
    if mask is not None:
        return [_apply_dropout(
            x, dropout_ratio, mask=mask[:x.shape[0]]) for x in xs]
    else:
        return [_apply_dropout(x, dropout_ratio) for x in xs]


def _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    if recurrent_dropout_ratio <= 0.0 and not use_variational_dropout:
        return F.connection.n_step_lstm.n_step_lstm_base(
            n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
            **kwargs)
    return _n_step_rnn_impl(
        F.connection.n_step_lstm._lstm,
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        recurrent_dropout_ratio, use_variational_dropout)


def n_step_lstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    return _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, False,
        recurrent_dropout_ratio, use_variational_dropout, **kwargs)


def n_step_bilstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs,
        recurrent_dropout_ratio=0.0, use_variational_dropout=False, **kwargs):
    return _n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, True,
        recurrent_dropout_ratio, use_variational_dropout, **kwargs)


class NStepRNNBase(chainer.links.connection.n_step_rnn.NStepRNNBase):

    def __init__(self, n_layers, in_size, out_size, dropout,
                 recurrent_dropout=0.0, use_variational_dropout=False,
                 **kwargs):
        self.recurrent_dropout = recurrent_dropout
        self.use_variational_dropout = use_variational_dropout
        super().__init__(n_layers, in_size, out_size, dropout, **kwargs)


class NStepLSTMBase(NStepRNNBase):
    n_weights = 8

    def forward(self, hx, cx, xs, **kwargs):
        (hy, cy), ys = self._call([hx, cx], xs, **kwargs)
        return hy, cy, ys


class NStepLSTM(NStepLSTMBase):
    use_bi_direction = False

    def rnn(self, *args):
        assert len(args) == 7
        return n_step_lstm(
            *args, self.recurrent_dropout, self.use_variational_dropout)

    @property
    def n_cells(self):
        return 2


class NStepBiLSTM(NStepLSTMBase):
    use_bi_direction = True

    def rnn(self, *args):
        assert len(args) == 7
        return n_step_bilstm(
            *args, self.recurrent_dropout, self.use_variational_dropout)

    @property
    def n_cells(self):
        return 2
