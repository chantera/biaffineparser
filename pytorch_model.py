import numpy as np
from teras.framework.pytorch.model import Biaffine, Embed, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mst


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
        indices = np.argsort(-np.array(lengths)).astype(np.int64)
        lengths = lengths[indices]
        X = torch.stack([X[idx] for idx in indices])
        X = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        R = self.blstm(X)[0]
        R = nn.utils.rnn.pad_packed_sequence(R, batch_first=True)[0]
        R = R.index_select(dim=0, index=_model_var(
            self, torch.from_numpy(np.argsort(indices).astype(np.int64))))
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
