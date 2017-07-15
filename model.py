import pickle
import re

from chainer import Chain
import chainer.functions as F
import numpy as np
from teras.dataset import Dataset
from teras.framework.chainer.model import Biaffine, BiGRU, BiLSTM, Embed, MLP
from teras.preprocessing import text

from utils import DEVELOP, read_conll


class BaselineParser(Chain):

    def __init__(self,
                 embeddings,
                 n_labels,
                 n_blstm_layers=3,
                 lstm_hidden_size=400,
                 use_gru=False,
                 n_arcmlp_layers=1,
                 n_arcmlp_units=500,
                 n_labelmlp_layers=1,
                 n_labelmlp_units=100,
                 dropout=0.33):
        embed_size = sum(embed.shape[1] for embed in embeddings)
        if lstm_hidden_size is None:
            lstm_hidden_size = embed_size
        super(BaselineParser, self).__init__()
        blstm_cls = BiGRU if use_gru else BiLSTM
        with self.init_scope():
            self.embed = Embed(*embeddings)
            self.blstm = blstm_cls(
                n_layers=n_blstm_layers,
                in_size=embed_size,
                out_size=lstm_hidden_size,
                dropout=dropout
            )
            activation = F.leaky_relu
            layers = [MLP.Layer(None, n_arcmlp_units, activation, dropout)
                      for i in range(n_arcmlp_layers)]
            self.mlp_arc_head = MLP(layers)
            layers = [MLP.Layer(None, n_arcmlp_units, activation, dropout)
                      for i in range(n_arcmlp_layers)]
            self.mlp_arc_dep = MLP(layers)
            layers = [MLP.Layer(None, n_labelmlp_units, activation, dropout)
                      for i in range(n_labelmlp_layers)]
            self.mlp_label_head = MLP(layers)
            layers = [MLP.Layer(None, n_labelmlp_units, activation, dropout)
                      for i in range(n_labelmlp_layers)]
            self.mlp_label_dep = MLP(layers)
            self.arc_biaffine = \
                Biaffine(n_arcmlp_units, n_arcmlp_units, 1,
                         nobias=(False, True, True))
            self.label_biaffine = \
                Biaffine(n_labelmlp_units, n_labelmlp_units, n_labels,
                         nobias=(False, False, False))
        self._embed_size = embed_size
        self._lstm_hidden_size = lstm_hidden_size
        self._dropout = dropout

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, word_tokens, pos_tokens):
        X = self.embed(word_tokens, pos_tokens)
        R = self.blstm(X)
        R = F.pad_sequence(R)
        H_arc_dep = self.mlp_arc_dep(R)
        H_arc_head = self.mlp_arc_head(R)
        arc_logits = self.arc_biaffine(H_arc_dep, H_arc_head)
        H_label_dep = self.mlp_label_dep(R)
        H_label_head = self.mlp_label_head(R)
        label_logits = self.label_biaffine(H_label_dep, H_label_head)
        # return arc_logits, label_logits
        return arc_logits


def compute_cross_entropy(y, t):
    y = F.reshape(y, y.shape[0:3])
    t = F.pad_sequence(t, padding=-1)
    loss = F.softmax_cross_entropy(y, t, ignore_label=-1)
    return loss


def compute_accuracy(y, t):
    y = F.reshape(y, y.shape[0:3])
    t = F.pad_sequence(t, padding=-1)
    accuracy = F.accuracy(y, t, ignore_label=-1)
    return accuracy


class DatasetProcessor:

    def __init__(self, word_embed_file, pos_embed_size):
        self.word_processor = \
            (text.Preprocessor(embed_file=word_embed_file, unknown="UNKNOWN")
             if not DEVELOP else
             text.Preprocessor(embed_file=None, embed_size=pos_embed_size))
        self.pos_processor = \
            text.Preprocessor(embed_file=None, embed_size=pos_embed_size)
        self.word_processor.set_preprocess_func(
            lambda x: re.sub(r'[0-9]', '0', x.lower()))

    def load(self, file, train=True):
        # [[words, postags, heads, labels, sentence], ...]
        samples = []

        sentences = read_conll(file)
        if not DEVELOP:
            # lookup words in pretrained vocabulary
            word_transform_one = self.word_processor.transform_one
        else:
            word_transform_one = self.word_processor.fit_transform_one \
                if train else self.word_processor.transform_one

        for i, sentence in enumerate(sentences):
            if DEVELOP and len(samples) >= 100:
                break
            words = [token['form'] for token in sentence]
            w_tokens = word_transform_one(words, preprocess=True)
            p_tokens = self.pos_processor.fit_transform_one(
                [token['postag'] for token in sentence], preprocess=True)
            heads = np.array([token['head'] for token in sentence],
                             dtype=np.int32)
            # labels = [token['deprel'] for token in sentence]
            # samples.append((w_tokens, p_tokens, heads,
            #                 labels, words))
            samples.append((w_tokens, p_tokens, heads))
        return Dataset(samples)

    @property
    def word_embeddings(self):
        return self.word_processor.get_embeddings()

    @property
    def pos_embeddings(self):
        return self.pos_processor.get_embeddings()

    def save_to(self, file):
        with open(file, mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_processor(file):
        instance = None
        with open(file, mode='rb') as f:
            instance = pickle.load(f)
        return instance
