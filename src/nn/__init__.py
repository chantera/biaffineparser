import chainer
import chainer.functions as F


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

    def __call__(self, x):
        return _EmbedIDFunction(x, self.W, self.ignore_label, self.fix_weight)


class _EmbedIDFunction(F.connection.embed_id.EmbedIDFunction):

    def __init__(self, ignore_label=None, fix_weight=False):
        super().__init__(ignore_label)
        self.fix_weight = fix_weight

    def backward(self, indexes, grad_outputs):
        if self.fix_weight:
            return None, None
        return super().backward(indexes, grad_outputs)


class MLP(chainer.link.ChainList):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        super().__init__(*layers)

    def __call__(self, x):
        for layer in self:
            x = layer(x)
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
            return dropout(x, self.dropout)


def dropout(x, ratio=.5, **kwargs):
    """Disable dropout when ratio == 0.0."""
    enabled = chainer.configuration.config.train and ratio > 0.0
    with chainer.using_config('train', enabled):
        return F.dropout(x, ratio, **kwargs)
