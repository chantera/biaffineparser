import chainer
import chainer.functions as F
import numpy as np

import nn


"""
NOTE: Weight Initialization comparison
  - A. This Implementation
    - URL: https://github.com/chantera/biaffineparser
    - Framework: Chainer (v5.2)
    - Initialization:
      - Embeddings (word):
      - Embeddings (postag):
      - BiLSTMs:
      - MLPs:
      - Biffines:
  - B. Original
    - URL: https://github.com/tdozat/Parser-v1
      <tree:0739216129cd39d69997d28cbc4133b360ea3934>
    - Framework: TensorFlow (<1.0, v0.8?)
      - See: https://github.com/tdozat/Parser-v1/issues/9
    - Initialization:
      - Embeddings (word): zero + pretrained (normalized with std)
      - Embeddings (postag): random_normal
      - BiLSTMs: W: orthonormal_initializer, b: zero
      - MLPs: W: orthonormal_initializer, b: zero
      - Biaffines: zero
  - C. GluonNLP
    - URL: https://github.com/dmlc/gluon-nlp
      <tree:8222eff32bf7bd08d04edfb8087f71836b523aec> (v0.5.0)
      - Path: scripts/parsing/
    - Framework: MXNet (latest: v1.5.0b)
    - Initialization:
      - Embeddings (word): zero + pretrained (normalized with std)
      - Embeddings (postag): random_normal
      - BiLSTMs: W: orthonormal_initializer, b: zero (forget gate bias: -1.0)
      - MLPs: W: orthonormal_initializer, b: zero
      - Biaffines: zero
  - D. StanfordNLP
    - URL: https://github.com/stanfordnlp/stanfordnlp
      <tree:0c4195017ee720cbd5d706fc59805648cd8f9dac> (v0.1.0)
      - Path: stanfordnlp/models/depparse/
    - Note: This is the implementation of the following paper:
            [http://aclweb.org/anthology/K18-2016]
    - Framework: PyTorch (v1.0)
    - Initialization: *They also used additional embeddings.
      - Embeddings (word): `torch.nn.Embeddings` default (N(0,1)) + pretrained
      - Embeddings (postag): `torch.nn.Embeddings` default
      - BiLSTMs: `torch.nn.LSTM` default
            (uniform(-s,s) for both W and b where s=sqrt(1/hidden_size))
      - MLPs: `torch.nn.Linear` default
            (uniform(-s,s) for both W and b where s=sqrt(1/fan_in))
      - Biaffines: zero
  - E. NeuroNLP2
    - URL: https://github.com/XuezheMax/NeuroNLP2
      <tree:fff2639ff3842c9fd0de91c8f6184f7d1d9f5d21>
    - Note: This is the implementation of the following paper:
            [http://www.aclweb.org/anthology/P18-1130]
    - Framework: PyTorch (v0.3)
    - Initialization: *They also used character embeddings.
      - Embeddings (word): pretrained only (no use of trainable embeddings)
      - Embeddings (postag): uniform(-s,s) where s=sqrt(3/dim)
      - BiLSTMs: W: uniform(-s,s) for where s=sqrt(1/hidden_size), b: zero
      - MLPs: `torch.nn.Linear` default
      - Biffines: U,W1,W2: xavier_uniform, b: zero
  - F. AllenNLP
    - URL: https://github.com/allenai/allennlp
      <tree:87f977a5fd58568d44d68499521ab0d71f8a0012> (v0.8.1)
      - Path: allennlp/models/biaffine_dependency_parser.py
    - Framework: PyTorch (v1.0 or v0.4.1)
    - Initialization: The configurations is not available.
"""


class BiaffineParser(chainer.Chain):

    def __init__(self, n_labels, encoder,
                 arc_mlp_units=500, rel_mlp_units=100,
                 arc_mlp_dropout=0.33, rel_mlp_dropout=0.33):
        super().__init__()
        if isinstance(arc_mlp_units, int):
            arc_mlp_units = [arc_mlp_units]
        if isinstance(rel_mlp_units, int):
            rel_mlp_units = [rel_mlp_units]
        mlp_activation = F.leaky_relu
        with self.init_scope():
            # This results in: uniform(-s, s) where s = sqrt(1/fan_in)
            # This is same as the Initialization of `torch.nn.Linear`
            # which is used in NeuroNLP2 [https://github.com/XuezheMax/NeuroNLP2]
            # and 
            # Note that orthonormal_initializer is used in the original implementation.
            uniform = chainer.initializers.HeUniform(scale=np.sqrt(1. / 6.))
            # self.encoder = encoder
            self.mlp_arc_head = nn.MLP([nn.MLP.Layer(
                None, u, mlp_activation, arc_mlp_dropout, initialW=uniform,
                initial_bias=uniform) for u in arc_mlp_units])
            self.mlp_arc_dep = nn.MLP([nn.MLP.Layer(
                None, u, mlp_activation, arc_mlp_dropout, initialW=uniform,
                initial_bias=uniform) for u in arc_mlp_units])
            self.mlp_rel_head = nn.MLP([nn.MLP.Layer(
                None, u, mlp_activation, rel_mlp_dropout, initialW=uniform,
                initial_bias=uniform) for u in rel_mlp_units])
            self.mlp_rel_dep = nn.MLP([nn.MLP.Layer(
                None, u, mlp_activation, rel_mlp_dropout, initialW=uniform,
                initial_bias=uniform) for u in rel_mlp_units])
            self.biaf_arc
            self.biaf_rel

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, words, pretrained_words, postags):
        hs = self.encoder(words, pretrained_words, postags)
        hs_arc_h, hs_arc_d = self.mlp_arc_dep(hs), self.mlp_arc_dep(hs)
        hs_rel_h, hs_rel_d = self.mlp_rel_dep(hs), self.mlp_rel_dep(hs)
        logits_arc, logits_rel = (self.biaf_arc(hs_arc_h, hs_arc_d),
                                  self.biaf_rel(hs_rel_h, hs_rel_d))
        return logits_arc, logits_rel

    def compute_loss(self, x, y):
        return chainer.Variable(np.zeros(1))


chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)


# hack
chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)


"""
def to_device(x, device=None):
    return chainer.dataset.convert.to_device(device, x)


def _update(optimizer, loss):
    optimizer.target.cleargrads()
    loss.backward()
    optimizer.update()


def chainer_train_on(*args, **kwargs):
    chainer.config.train = True
    chainer.config.enable_backprop = True


def chainer_train_off(*args, **kwargs):
    chainer.config.train = False
    chainer.config.enable_backprop = False


def set_seed(seed):
    seed = str(seed)
    os.environ['CHAINER_SEED'] = seed
    os.environ['CUPY_SEED'] = seed


def set_debug(debug):
    if debug:
        chainer.config.debug = True
        chainer.config.type_check = True
    else:
        chainer.config.debug = False
        chainer.config.type_check = False


def set_model_to_device(model, device_id=-1):
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        model.to_gpu()
    else:
        model.to_cpu()


set_debug(chainer.config.debug)
chainer.config.use_cudnn = 'auto'


config = {
    'update': _update,
    'hooks': {
        TrainEvent.EPOCH_TRAIN_BEGIN: chainer_train_on,
        TrainEvent.EPOCH_VALIDATE_BEGIN: chainer_train_off,
    },
    'callbacks': []
}
"""
