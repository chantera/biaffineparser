import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import mst


class BiaffineParser(nn.Module):

    def __init__(self, n_rels, encoder,
                 arc_mlp_units=500, rel_mlp_units=100,
                 arc_mlp_dropout=0.0, rel_mlp_dropout=0.0):
        super().__init__()
        if isinstance(arc_mlp_units, int):
            arc_mlp_units = [arc_mlp_units]
        if isinstance(rel_mlp_units, int):
            rel_mlp_units = [rel_mlp_units]

        def _create_mlp(in_size, units, dropout):
            return MLP([MLP.Layer(
                units[i - 1] if i > 0 else in_size, u,
                lambda x: F.leaky_relu(x, negative_slope=0.1), dropout)
                        for i, u in enumerate(units)])

        self.encoder = encoder
        h_dim = self.encoder.out_size
        self.mlp_arc_head = _create_mlp(h_dim, arc_mlp_units, arc_mlp_dropout)
        self.mlp_arc_dep = _create_mlp(h_dim, arc_mlp_units, arc_mlp_dropout)
        self.mlp_rel_head = _create_mlp(h_dim, rel_mlp_units, rel_mlp_dropout)
        self.mlp_rel_dep = _create_mlp(h_dim, rel_mlp_units, rel_mlp_dropout)
        self.biaf_arc = Biaffine(arc_mlp_units[-1], arc_mlp_units[-1], 1)
        self.biaf_rel = Biaffine(rel_mlp_units[-1], rel_mlp_units[-1], n_rels)
        self._results = {}

    def forward(self, words, pretrained_words, postags, *args):
        self._results.clear()
        # [n; B], [n; B], [n; B] => (B, n_max, d)
        hs = self.encode(words, pretrained_words, postags)
        hs_arc_h = self.mlp_arc_head(hs)
        hs_arc_d = self.mlp_arc_dep(hs)
        hs_rel_h = self.mlp_rel_head(hs)
        hs_rel_d = self.mlp_rel_dep(hs)
        logits_arc = self.biaf_arc(hs_arc_d, hs_arc_h).squeeze_(3)
        mask = _mask_arc(
            logits_arc, self._lengths, mask_loop=not(self.training))
        self._mask = mask = _from_numpy(mask, np.bool, logits_arc.is_cuda)
        self._logits_arc = logits_arc.masked_fill_(mask.logical_not(), -1e8)
        self._logits_rel = self.biaf_rel(hs_rel_d, hs_rel_h)
        # => (B, n_max, n_max), (B, n_max, n_max, n_rels)
        return self._logits_arc, self._logits_rel

    def encode(self, *args):
        self._hs, self._lengths = self.encoder(*args[:3])
        return self._hs

    def parse(self, words, pretrained_words, postags, use_cache=True):
        if len(self._results) == 0 or not use_cache:
            self.forward(words, pretrained_words, postags)
        arcs = _parse_by_graph(self._logits_arc, self._lengths, self._mask)
        rels = _decode_rels(self._logits_rel, arcs, self._lengths)
        arcs = [arcs_i[:n] for arcs_i, n in zip(arcs, self._lengths)]
        rels = [rels_i[:n] for rels_i, n in zip(rels, self._lengths)]
        parsed = list(zip(arcs, rels))
        return parsed

    def compute_loss(self, y, t):
        self._results = _compute_metrics(y, t, self._lengths, False)
        return self._results['arc_loss'] + self._results['rel_loss']

    def compute_accuracy(self, y, t, use_cache=True):
        arc_accuracy = self._results.get('arc_accuracy', None)
        rel_accuracy = self._results.get('rel_accuracy', None)
        if not use_cache or (arc_accuracy is None and rel_accuracy is None):
            results = _compute_metrics(y, t, self._lengths, False)
            arc_accuracy = results.get('arc_accuracy', None)
            rel_accuracy = results.get('rel_accuracy', None)
            self._results.update({
                'arc_accuracy': arc_accuracy,
                'rel_accuracy': rel_accuracy,
            })
        return arc_accuracy, rel_accuracy


def _mask_arc(logits_arc, lengths, mask_loop=True):
    mask = np.zeros(logits_arc.shape, dtype=np.int32)
    for i, length in enumerate(lengths):
        mask[i, :length, :length] = 1
    if mask_loop:
        mask *= (1 - np.eye(logits_arc.shape[2], dtype=np.int32))
    return mask


def _parse_by_graph(logits_arc, lengths, mask=None):
    probs = F.softmax(logits_arc, dim=2).detach()
    if mask is not None:
        probs.mul_(mask)
    probs = probs.cpu().numpy()
    trees = np.full((len(lengths), max(lengths)), -1, dtype=np.int32)
    for i, (probs_i, length) in enumerate(zip(probs, lengths)):
        trees[i, 1:length] = mst.mst(probs_i[:length, :length])[0][1:]
    return trees


def _decode_rels(logits_rel, trees, lengths, root=0):
    steps = np.arange(trees.shape[1])
    logits_rel = [logits_rel[i, steps, arcs] for i, arcs in enumerate(trees)]
    logits_rel = torch.stack(logits_rel, dim=0).detach()
    logits_rel[:, :, root] = -1e8
    rels = logits_rel.argmax(dim=2)
    rels = rels.cpu().numpy()
    for rels_i, arcs_i in zip(rels, trees):
        rels_i[:] = np.where(arcs_i == 0, root, rels_i)
    rels[:, 0] = -1
    return rels


def _compute_metrics(parsed, gold_batch, lengths,
                     use_predicted_arcs_for_rels=True):
    logits_arc, logits_rel, *_ = parsed
    true_arcs, true_rels, *_ = zip(*gold_batch)

    # exclude attachment from the root
    logits_arc, logits_rel = logits_arc[:, 1:], logits_rel[:, 1:]
    true_arcs = _from_numpy(_np_pad_sequence(true_arcs, padding=-1)[:, 1:],
                            dtype=np.int64, cuda=logits_arc.is_cuda)
    true_rels = _from_numpy(_np_pad_sequence(true_rels, padding=-1)[:, 1:],
                            dtype=np.int64, cuda=logits_rel.is_cuda)
    lengths = lengths - 1

    b, n_deps, n_heads = logits_arc.shape
    logits_arc_flatten = logits_arc.contiguous().view(b * n_deps, n_heads)
    true_arcs_flatten = true_arcs.contiguous().view(b * n_deps)
    arc_loss = F.cross_entropy(logits_arc_flatten, true_arcs_flatten,
                               ignore_index=-1, reduction='sum')
    arc_loss.div_(lengths.sum())
    arc_accuracy = _accuracy(
        logits_arc_flatten, true_arcs_flatten, ignore_index=-1)

    if use_predicted_arcs_for_rels:
        parsed_arcs = logits_arc.argmax(dim=2)
    else:
        parsed_arcs = true_arcs.masked_fill(true_arcs == -1, 0)
    b, n_deps, n_heads, n_rels = logits_rel.shape
    logits_rel = logits_rel.gather(dim=2, index=parsed_arcs.view(
        *parsed_arcs.size(), 1, 1).expand(-1, -1, -1, n_rels))
    logits_rel_flatten = logits_rel.contiguous().view(b * n_deps, n_rels)
    true_rels_flatten = true_rels.contiguous().view(b * n_deps)
    rel_loss = F.cross_entropy(logits_rel_flatten, true_rels_flatten,
                               ignore_index=-1, reduction='sum')
    rel_loss.div_(lengths.sum())
    rel_accuracy = _accuracy(
        logits_rel_flatten, true_rels_flatten, ignore_index=-1)

    return {'arc_loss': arc_loss, 'arc_accuracy': arc_accuracy,
            'rel_loss': rel_loss, 'rel_accuracy': rel_accuracy}


def _accuracy(y, t, ignore_index=None):
    pred = y.argmax(dim=1)
    if ignore_index is not None:
        mask = (t == ignore_index)
        ignore_cnt = mask.sum()
        ignore = torch.tensor([ignore_index], dtype=torch.int64)
        if pred.is_cuda:
            ignore = ignore.cuda()
        pred = torch.where(mask, ignore, pred)
        count = (pred == t).sum() - ignore_cnt
        total = t.numel() - ignore_cnt
    else:
        count = (pred == t).sum()
        total = t.numel()
    return count.item(), total.item()


def _np_pad_sequence(xs, padding=0):
    length = max(len(x) for x in xs)
    shape = (len(xs), length) + xs[0].shape[1:]
    y = np.empty(shape, xs[0].dtype)
    if length == 0:
        return y
    for i, x in enumerate(xs):
        n = len(x)
        if n == length:
            y[i] = x
        else:
            y[i, 0:n] = x
            y[i, n:] = padding
    return y


class Encoder(nn.Module):

    def __init__(self,
                 word_embeddings,
                 pretrained_word_embeddings=None,
                 postag_embeddings=None,
                 n_lstm_layers=3,
                 lstm_hidden_size=None,
                 embeddings_dropout=0.0,
                 lstm_dropout=0.0,
                 recurrent_dropout=0.0):
        super().__init__()
        self._use_pretrained_word = self._use_postag = False
        embeddings = [(word_embeddings, False)]  # (weights, fixed)
        lstm_in_size = word_embeddings.shape[1]
        if pretrained_word_embeddings is not None:
            embeddings.append((pretrained_word_embeddings, True))
            self._use_pretrained_word = True
        if postag_embeddings is not None:
            embeddings.append((postag_embeddings, False))
            lstm_in_size += postag_embeddings.shape[1]
            self._use_postag = True
        self.embeds = nn.ModuleList(nn.Embedding.from_pretrained(
            torch.from_numpy(w), fixed) for w, fixed in embeddings)
        if lstm_hidden_size is None:
            lstm_hidden_size = lstm_in_size
        # TODO: Implement recurrent dropout
        self.bilstm = nn.LSTM(
            lstm_in_size, lstm_hidden_size, n_lstm_layers,
            batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.embeddings_dropout = SequenceDropout(embeddings_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self._hidden_size = lstm_hidden_size

    def forward(self, *xs):
        # [(n, d_word); B], [(n, d_word); B], [(n, d_pos); B]
        lengths = np.array([x.size for x in xs[0]], dtype=np.int32)
        xs_flatten = (np.concatenate(xs_each, axis=0) for xs_each in xs)
        rs = [embed(_from_numpy(xs_each, np.int64, embed.weight.is_cuda))
              for embed, xs_each in zip(self.embeds, xs_flatten)]
        rs = self._concat_embeds(rs)
        # => [(n, d_word + d_pos); B]
        if np.all(lengths == lengths[0]):
            sections = int(lengths[0])
        else:
            sections = tuple(lengths)
        rs = torch.split(self.lstm_dropout(rs), sections, dim=0)
        rs = nn.utils.rnn.pack_sequence(rs, enforce_sorted=False)
        hs, _ = self.bilstm(rs)
        hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)
        hs = self.lstm_dropout(hs)
        return hs, lengths

    def _concat_embeds(self, embed_outputs):
        rs_postags = embed_outputs.pop() if self._use_postag else None
        rs_words_pretrained = embed_outputs.pop() \
            if self._use_pretrained_word else None
        rs_words = embed_outputs.pop()
        if rs_words_pretrained is not None:
            rs_words += rs_words_pretrained
        rs = [rs_words]
        if rs_postags is not None:
            rs.append(rs_postags)
        rs = self.embeddings_dropout(rs)
        rs = torch.cat(rs, dim=1) if len(rs) > 1 else rs[0]
        return rs

    @property
    def out_size(self):
        return self._hidden_size * 2


def _from_numpy(x, dtype=None, cuda=False):
    if dtype is not None:
        x = x.astype(dtype)
    x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return x


class SequenceDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, xs):
        return _embed_dropout(xs, self.p, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.p)


def _embed_dropout(xs, p=0.5, training=True):
    """
    Drop representations with scaling.
    https://github.com/tdozat/Parser-v2/blob/304c638aa780a5591648ef27060cfa7e4bee2bd0/parser/neural/models/nn.py#L50  # NOQA
    """
    if not training or p == 0.0:
        return xs
    masks = (np.random.rand(len(xs), xs[0].size(0)) >= p).astype(np.float32)
    scale = len(masks) / np.maximum(np.sum(masks, axis=0, keepdims=True), 1)
    masks = np.expand_dims(masks * scale, axis=2)
    ys = [xs_each * _from_numpy(mask, cuda=xs_each.is_cuda)
          for xs_each, mask in zip(xs, masks)]
    return ys


class MLP(nn.Sequential):

    def __init__(self, layers):
        assert all(isinstance(layer, MLP.Layer) for layer in layers)
        super().__init__(*layers)

    class Layer(nn.Module):

        def __init__(self, in_size, out_size=None,
                     activation=None, dropout=0.0, bias=True):
            super().__init__()
            if activation is not None and not callable(activation):
                raise TypeError("activation must be callable: type={}"
                                .format(type(activation)))
            self.linear = nn.Linear(in_size, out_size, bias)
            self.activate = activation
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h = self.linear(x)
            if self.activate is not None:
                h = self.activate(h)
            return self.dropout(h)


class Biaffine(nn.Module):

    def __init__(self, in1_features, in2_features, out_features):
        super().__init__()
        self.bilinear = PairwiseBilinear(
            in1_features + 1, in2_features + 1, out_features)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)],
                           dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)],
                           dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # NOQA
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(in1_features, out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (B * n1, d1) @ (d1, O * d2) => (B * n1, O * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (B, n1 * O, d2) @ (B, d2, n2) => (B, n1 * O, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        # => (B, n1, n2, O)
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}' \
            .format(self.in1_features, self.in2_features, self.out_features,
                    self.bias is not None)
