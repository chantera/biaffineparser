import re

from chainer import Chain
import chainer.functions as F
import numpy as np
from teras.dataset.loader import CorpusLoader
from teras.framework.chainer.model import Biaffine, BiGRU, BiLSTM, Embed, MLP
from teras.io.reader import ConllReader
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

    def __init__(self, model):
        self.model = model

    def forward(self, word_tokens, pos_tokens):
        lengths = [len(tokens) for tokens in word_tokens]
        arc_logits, label_logits = self.model.forward(word_tokens, pos_tokens)
        pred_arcs = self.model.xp.argmax(arc_logits.data, axis=1)
        label_logits = F.transpose(label_logits, (0, 2, 1, 3))
        label_logits = [_logits[np.arange(_length), _arcs[:_length]]
                        for _logits, _arcs, _length
                        in zip(label_logits, pred_arcs, lengths)]
        label_logits = F.pad_sequence(label_logits)
        return arc_logits, label_logits

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
