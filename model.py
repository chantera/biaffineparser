import re

from chainer import Chain, Variable
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
        # arc_logits = F.reshape(arc_logits, arc_logits.shape[0:3])
        arc_logits = F.squeeze(arc_logits, axis=3)
        H_label_dep = self.mlp_label_dep(R)
        H_label_head = self.mlp_label_head(R)
        label_logits = self.label_biaffine(H_label_dep, H_label_head)
        return arc_logits, label_logits


class BiaffineParser(object):

    def __init__(self, model):
        self.model = model

    def forward(self, word_tokens, pos_tokens):
        xp = self.model.xp
        lengths = [len(tokens) for tokens in word_tokens]
        # print(lengths)
        # mask = F.pad_sequence(word_tokens, padding=-2)
        # mask = (mask.data == -2) * -1
        # print(mask)
        arc_logits, label_logits = self.model.forward(word_tokens, pos_tokens)
        b, l1, l2, d = label_logits.shape
        pred_arcs = xp.argmax(arc_logits.data, axis=1)
        # label_logits = F.reshape(F.transpose(label_logits, (0, 2, 1, 3)), (b, l2, l1 * d))
        label_logits = F.transpose(label_logits, (0, 2, 1, 3))
        label_logits = [_logits[np.arange(_length), _arcs[:_length]] for _logits, _arcs, _length in zip(label_logits, pred_arcs, lengths)]
        # label_logits = [_logits[np.arange(_length), _arcs[:_length]] for _logits, _arcs, _length in zip(label_logits, pred_arcs, lengths)]
        label_logits = F.pad_sequence(label_logits)
        # label_logits = F.transpose(label_logits, (0, 2, 1, 3))
        print(label_logits.shape)
        # label_logits = F.squeeze(label_logits, axis=1)
        # print(label_logits.shape)
        #
        # v = F.select_item(label_logits, pred_arcs)
        # print(v.shape)
        # pred_arcs = F.argmax(arc_logits, axis=1)
        # b, n = pred_arcs.shape
        # print(pred_arcs.shape)
        # pred_arcs = F.embed_id(pred_arcs + mask,
        #                        xp.ones((n, 1), xp.float32),
        #                        ignore_label=-1)
        # # print(pred_arcs)
        """
        pred_arcs = xp.argmax(arc_logits.data, axis=1)
        # onehot = Variable(xp.zeros(label_logits.shape, xp.float32))
        b, l1, l2, d = label_logits.shape
        onehot = xp.zeros(label_logits.shape, xp.float32)
        for _onehot, _arcs, _length in zip(onehot, pred_arcs, lengths):
                # zip(onehot, pred_arcs, lengths):
            # for i, _arc in enumerate(_arcs):
            #     _onehot[i, ]
            # print(_arcs[:_length])
            # print(_arcs[:_length].shape)
            # assert False
            # _onehot[:_length, _arcs[:_length].T] = 1
            _onehot[xp.arange(_length), _arcs[:_length]] = 1
            # _onehot += xp.take(, _arcs[:_length], axis=1)[:_length] = 1
        label_logits = F.reshape(F.transpose(label_logits, (0, 3, 1, 2)),
                                 (b * d, l1, l2))
        onehot = F.reshape(F.transpose(onehot, (0, 3, 2, 1)),
                           (b * d, l2, l1))
        label_logits = F.batch_matmul(label_logits, onehot)
        label_logits = F.transpose(F.reshape(label_logits, (b, d, l1, l2)),
                                   (0, 2, 3, 1))
        label_logits = F.reshape(label_logits, (b, l1, d))
        print(label_logits)


        """

        return arc_logits, label_logits
        # print(label_logits.shape)
        # print(pred_arcs[0, 0], label_logits[0, 0])
        # print(label_logits.shape)
        # for o, a in zip(onehot[0], pred_arcs[0]):
        #     print(a, o)
        # print(pred_arcs[0], onehot[0])
            # print(_onehot[:_length].shape, _arcs[:_length])
        #     for _matrix, _arc in zip(_onehot, _arcs[_length]):
        #         _
        #     # print(_arcs[:_length].data)
        #     # _onehot[_arcs[:_length].data] = 1
        #     # print(_onehot)
        # print(pred_arcs[0], onehot[0])

        # print(pred_arcs)
        # b, n = pred_arcs.shape
        # for _logits, _arcs in zip(label_logits, pred_arcs):
        #     # for _logit, _arc in zip(_logits, _arcs):
        #     print(F.select_item(_logits, _arcs))
        # print(F.select_item(label_logits, pred_arcs))
        # label_logits = F.get_item(label_logits, (slice(None, None, None), slice(None, None, None), pred_arcs))
        # print(label_logits.shape)
        # arcs = xp.argmax(arc_logits.data, axis=1)
        # b = xp.zeros(label_logits.shape[:-1], xp.float32)
        # for _arcs in arcs:
        #     print(_arcs)
        # b[:, arcs] = 1
        # print(b[0])
        """
        print(pred_arcs[0])
        pred_arcs = F.embed_id(pred_arcs + mask,
                               xp.ones((n, 1), xp.float32),
                               ignore_label=-1)
        print(pred_arcs[0])
        pred_arcs = F.expand_dims(F.transpose(pred_arcs, (0, 2, 1)), axis=3)
        # print(pred_arcs[0,0])
        # print(label_logits.shape, .shape)
        label_logits = F.batch_matmul(label_logits, pred_arcs)
        # print(label_logits)
        #
        # label_logits = F.concat(F.embed_id() for _arcs, _logits]
        # v = F.embed_id(pred_arcs, F.reshape(label_logits, (b, n, -1)), ignore_label=-1)
        """


def compute_cross_entropy(y, t):
    arc_logits, label_logits = y
    true_arcs, true_labels = t.T

    # arc_logits = F.reshape(arc_logits, arc_logits.shape[0:3])
    b, l1, l2 = arc_logits.shape
    # print(arc_logits.shape)
    true_arcs = F.pad_sequence(true_arcs, padding=-1)
    # print(true_arcs.shape)
    arc_loss = F.softmax_cross_entropy(
        F.reshape(arc_logits, (b * l1, l2)),
        F.reshape(true_arcs, (b * l1,)),
        ignore_label=-1)

    # print(true_arcs.shape, pred_arcs.shape)
    # print(pred_arcs * label_logits)

    b, l1, d = label_logits.shape
    # label_logits = F.reshape(label_logits, (b * l1, d))
    # print(label_logits.shape)
    true_labels = F.pad_sequence(true_labels, padding=-1)
    # true_labels = F.reshape(true_labels, (b, l1))
    # print(true_labels.shape)
    print(true_labels)
    label_loss = F.softmax_cross_entropy(
        F.reshape(label_logits, (b * l1, d)),
        F.reshape(true_labels, (b * l1,)),
        ignore_label=-1)

    loss = arc_loss + label_loss
    return loss


def compute_accuracy(y, t):
    arc_logits, label_logits = y
    true_arcs, true_labels = t.T
    arc_logits = F.reshape(arc_logits, arc_logits.shape[0:3])
    true_arcs = F.pad_sequence(true_arcs, padding=-1)
    accuracy = F.accuracy(arc_logits, true_arcs, ignore_label=-1)
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

    def load(self, file, train=False):
        if train and not self.use_pretrained:
            # assign an index if the given word is not in vocabulary
            word_transform_one = self.get_processor('word').fit_transform_one
        else:
            # return the unknown word index if the word is not in vocabulary
            word_transform_one = self.get_processor('word').transform_one
        self._word_transform_one = word_transform_one
        return super(DataLoader, self).load(file, train)
