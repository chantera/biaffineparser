import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.chuliu_edmonds import chuliu_edmonds_one_root


def build_model(**kwargs) -> "BiaffineParser":
    embeddings = [
        (kwargs.get(f"{name}_vocab_size", 1), kwargs.get(f"{name}_embed_size", 100))
        for name in ["word", "pretrained_word", "postag"]
    ]
    if kwargs.get("pretrained_word_embeddings") is not None:
        embeddings[1] = kwargs["pretrained_word_embeddings"]
    dropout_ratio = kwargs.get("dropout", 0.33)
    encoder = BiLSTMEncoder(
        embeddings,
        reduce_embeddings=[0, 1],
        n_lstm_layers=kwargs.get("n_lstm_layers", 3),
        lstm_hidden_size=kwargs.get("lstm_hidden_size", 400),
        embedding_dropout=kwargs.get("embedding_dropout", dropout_ratio),
        lstm_dropout=kwargs.get("lstm_dropout", dropout_ratio),
        recurrent_dropout=kwargs.get("recurrent_dropout", dropout_ratio),
    )
    model = BiaffineParser(
        encoder,
        n_rels=kwargs.get("n_rels"),
        arc_mlp_units=kwargs.get("arc_mlp_units", 500),
        rel_mlp_units=kwargs.get("rel_mlp_units", 100),
        arc_mlp_dropout=kwargs.get("arc_mlp_dropout", dropout_ratio),
        rel_mlp_dropout=kwargs.get("rel_mlp_dropout", dropout_ratio),
    )
    return model


class BiaffineParser(nn.Module):
    def __init__(
        self,
        encoder: "Encoder",
        n_rels: Optional[int] = None,
        arc_mlp_units: Union[Sequence[int], int] = 500,
        rel_mlp_units: Union[Sequence[int], int] = 100,
        arc_mlp_dropout: float = 0.0,
        rel_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        assert n_rels is not None  # TODO: support unlabeled parsing
        if isinstance(arc_mlp_units, int):
            arc_mlp_units = [arc_mlp_units]
        if isinstance(rel_mlp_units, int):
            rel_mlp_units = [rel_mlp_units]

        def _create_mlp(in_size, units, dropout):
            return MLP(
                [
                    MLP.Layer(
                        units[i - 1] if i > 0 else in_size,
                        u,
                        lambda x: F.leaky_relu(x, negative_slope=0.1),
                        dropout,
                    )
                    for i, u in enumerate(units)
                ]
            )

        self.encoder = encoder
        h_dim = self.encoder.out_size
        self.mlp_arc_head = _create_mlp(h_dim, arc_mlp_units, arc_mlp_dropout)
        self.mlp_arc_dep = _create_mlp(h_dim, arc_mlp_units, arc_mlp_dropout)
        self.mlp_rel_head = _create_mlp(h_dim, rel_mlp_units, rel_mlp_dropout)
        self.mlp_rel_dep = _create_mlp(h_dim, rel_mlp_units, rel_mlp_dropout)
        self.biaf_arc = Biaffine(arc_mlp_units[-1], arc_mlp_units[-1], 1)
        self.biaf_rel = Biaffine(rel_mlp_units[-1], rel_mlp_units[-1], n_rels)

    def forward(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        hs, lengths = self.encoder(*input_ids)  # => (b, n, d)
        hs_arc_h = self.mlp_arc_head(hs)
        hs_arc_d = self.mlp_arc_dep(hs)
        hs_rel_h = self.mlp_rel_head(hs)
        hs_rel_d = self.mlp_rel_dep(hs)
        logits_arc = self.biaf_arc(hs_arc_d, hs_arc_h).squeeze_(3)
        mask = _mask_arc(lengths, mask_loop=not self.training)
        if mask is not None:
            logits_arc.masked_fill_(mask.logical_not().to(logits_arc.device), -float("inf"))
        logits_rel = self.biaf_rel(hs_rel_d, hs_rel_h)
        return logits_arc, logits_rel, lengths  # (b, n, n), (b, n, n, n_rels), (b,)

    def parse(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.decode(self.forward(input_ids))

    @torch.no_grad()
    def decode(
        self,
        logits_arc: torch.Tensor,
        logits_rel: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        arcs = _parse_graph(logits_arc, lengths)
        rels = _decode_rels(logits_rel, arcs)
        return arcs, rels

    def compute_metrics(
        self,
        logits_arc: torch.Tensor,
        logits_rel: Optional[torch.Tensor],
        true_arcs: Sequence[torch.Tensor],
        true_rels: Optional[Sequence[torch.Tensor]],
    ) -> Dict[str, Any]:
        true_arcs = nn.utils.rnn.pad_sequence(true_arcs, batch_first=True, padding_value=-1)
        if true_rels is not None:
            true_rels = nn.utils.rnn.pad_sequence(true_rels, batch_first=True, padding_value=-1)
        result = _compute_metrics(logits_arc, true_arcs, logits_rel, true_rels, ignore_index=-1)
        result["loss"] = result["arc_loss"] + (result["rel_loss"] or 0.0)
        return result


@torch.no_grad()
def _mask_arc(lengths: torch.Tensor, mask_loop: bool = True) -> Optional[torch.Tensor]:
    b, n = lengths.numel(), lengths.max()
    if torch.all(lengths == n):
        if not mask_loop:
            return None
        mask = torch.ones(b, n, n)
    else:
        mask = torch.zeros(b, n, n)
        for i, length in enumerate(lengths):
            mask[i, :length, :length] = 1
    if mask_loop:
        mask.masked_fill_(torch.eye(n, dtype=torch.bool), 0)
    return mask


@torch.no_grad()
def _parse_graph(
    logits_arc: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    mask = mask or _mask_arc(lengths, mask_loop=True)
    probs = (F.softmax(logits_arc, dim=2) * mask.to(logits_arc.device)).cpu().numpy()
    trees = torch.full((lengths.numel(), max(lengths)), -1)
    for i, length in enumerate(lengths):
        trees[i, :length] = torch.from_numpy(chuliu_edmonds_one_root(probs[i, :length, :length]))
    trees[:, 0] = -1
    return trees


@torch.no_grad()
def _decode_rels(logits_rel: torch.Tensor, trees: torch.Tensor, root: int = 0) -> torch.Tensor:
    steps = torch.arange(trees.size(1))
    logits_rel = [logits_rel[i, steps, heads] for i, heads in enumerate(trees)]
    logits_rel = torch.stack(logits_rel, dim=0)
    logits_rel[:, :, root] = -float("inf")
    rels = logits_rel.argmax(dim=2).detach().cpu()
    rels.masked_fill_(trees == 0, root)
    return rels


def _compute_metrics(
    logits_arc: torch.Tensor,
    true_arcs: torch.Tensor,
    logits_rel: Optional[torch.Tensor] = None,
    true_rels: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = -1,
    use_predicted_arcs_for_rels: bool = False,
) -> Dict[str, Any]:
    result = dict.fromkeys(["arc_loss", "arc_accuracy", "rel_loss", "rel_accuracy"])

    def metrics(y, t, ignore_index):
        loss = F.cross_entropy(y, t, ignore_index=ignore_index, reduction="sum")
        loss.div_((t != -1).sum())
        accuracy = categorical_accuracy(y, t, ignore_index)
        return loss, accuracy

    logits_arc, true_arcs = logits_arc[:, 1:], true_arcs[:, 1:]  # exclude attachment for the root
    logits_arc_flatten = logits_arc.contiguous().view(-1, logits_arc.size(-1))
    true_arcs_flatten = true_arcs.contiguous().view(-1)
    arc_loss, arc_accuracy = metrics(logits_arc_flatten, true_arcs_flatten, ignore_index)
    result.update(arc_loss=arc_loss, arc_accuracy=arc_accuracy)

    if logits_rel is None:
        return result
    elif true_rels is None:
        raise ValueError("'true_rels' must be given to compute loss for rels")

    if use_predicted_arcs_for_rels:
        arcs = logits_arc.argmax(dim=2)
    else:
        arcs = true_arcs.masked_fill(true_arcs == -1, 0)

    logits_rel, true_rels = logits_rel[:, 1:], true_rels[:, 1:]  # exclude attachment for the root
    gather_index = arcs.view(*arcs.size(), 1, 1).expand(-1, -1, -1, logits_rel.size(-1))
    logits_rel = torch.gather(logits_rel, dim=2, index=gather_index)
    logits_rel_flatten = logits_rel.contiguous().view(-1, logits_rel.size(-1))
    true_rels_flatten = true_rels.contiguous().view(-1)
    rel_loss, rel_accuracy = metrics(logits_rel_flatten, true_rels_flatten, ignore_index)
    result.update(rel_loss=rel_loss, rel_accuracy=rel_accuracy)

    return result


@torch.no_grad()
def categorical_accuracy(
    y: torch.Tensor, t: torch.Tensor, ignore_index: Optional[int] = None
) -> Tuple[int, int]:
    pred = y.argmax(dim=1)
    if ignore_index is not None:
        mask = t == ignore_index
        ignore_cnt = mask.sum()
        pred.masked_fill_(mask, ignore_index)
        count = (pred == t).sum() - ignore_cnt
        total = t.numel() - ignore_cnt
    else:
        count = (pred == t).sum()
        total = t.numel()
    return count.item(), total.item()


class Encoder(nn.Module):
    def forward(self, *input_ids: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the encoded sequences and their lengths."""
        raise NotImplementedError


class BiLSTMEncoder(Encoder):
    def __init__(
        self,
        embeddings: Iterable[Union[torch.Tensor, Tuple[int, int]]],
        reduce_embeddings: Optional[Sequence[int]] = None,
        n_lstm_layers: int = 3,
        lstm_hidden_size: Optional[int] = None,
        embedding_dropout: float = 0.0,
        lstm_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.embeds = nn.ModuleList()
        for item in embeddings:
            if isinstance(item, tuple):
                size, dim = item
                emb = nn.Embedding(size, dim)
            else:
                emb = nn.Embedding.from_pretrained(item, freeze=True)
            self.embeds.append(emb)
        self._reduce_embs = sorted(reduce_embeddings or [])

        embed_dims = [emb.weight.size(1) for emb in self.embeds]
        lstm_in_size = sum(embed_dims)
        if len(self._reduce_embs) > 1:
            lstm_in_size -= embed_dims[self._reduce_embs[0]] * (len(self._reduce_embs) - 1)
        if lstm_hidden_size is None:
            lstm_hidden_size = lstm_in_size
        self.bilstm = LSTM(
            lstm_in_size,
            lstm_hidden_size,
            n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.embedding_dropout = EmbeddingDropout(embedding_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self._hidden_size = lstm_hidden_size

    def forward(self, *input_ids: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(input_ids) != len(self.embeds):
            raise ValueError(f"exact {len(self.embeds)} types of sequences must be given")
        lengths = torch.tensor([x.size(0) for x in input_ids[0]])
        xs = [emb(torch.cat(ids_each, dim=0)) for emb, ids_each in zip(self.embeds, input_ids)]
        if len(self._reduce_embs) > 1:
            xs += [torch.sum(torch.stack([xs.pop(i) for i in reversed(self._reduce_embs)]), dim=0)]
        seq = self.lstm_dropout(self.embedding_dropout(torch.stack(xs, dim=1)))  # (B * n, emb, d)

        if torch.all(lengths == lengths[0]):
            hs, _ = self.bilstm(seq.view(len(lengths), lengths[0], -1))
        else:
            seq = torch.split(seq.view(seq.size(0), -1), tuple(lengths), dim=0)
            seq = nn.utils.rnn.pack_sequence(seq, enforce_sorted=False)
            hs, _ = self.bilstm(seq)
            hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)
        return self.lstm_dropout(hs), lengths

    @property
    def out_size(self) -> int:
        return self._hidden_size * 2


class EmbeddingDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Drop embeddings with scaling.
        https://github.com/tdozat/Parser-v2/blob/304c638aa780a5591648ef27060cfa7e4bee2bd0/parser/neural/models/nn.py#L50  # NOQA
        """
        if not self.training or self.p == 0.0:
            return xs
        with torch.no_grad():
            mask = torch.rand(xs.size()[:-1], device=xs.device) >= self.p  # (..., n_channels)
            scale = mask.size(-1) / torch.clamp(mask.sum(dim=-1, keepdims=True), min=1.0)
            mask = (mask * scale)[..., None]
        return xs * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


class LSTM(nn.LSTM):
    """LSTM with DropConnect."""

    __constants__ = nn.LSTM.__constants__ + ["recurrent_dropout"]

    def __init__(self, *args, **kwargs):
        self.recurrent_dropout = float(kwargs.pop("recurrent_dropout", 0.0))
        super().__init__(*args, **kwargs)

    def forward(self, input, hx=None):
        if not self.training or self.recurrent_dropout == 0.0:
            return super().forward(input, hx)
        __flat_weights = self._flat_weights
        p = self.recurrent_dropout
        self._flat_weights = [
            F.dropout(w, p) if name.startswith("weight_hh_") else w
            for w, name in zip(__flat_weights, self._flat_weights_names)
        ]
        self.flatten_parameters()
        ret = super().forward(input, hx)
        self._flat_weights = __flat_weights
        return ret


class MLP(nn.Sequential):
    def __init__(self, layers: Iterable["MLP.Layer"]):
        if not all(isinstance(layer, MLP.Layer) for layer in layers):
            raise TypeError("each layer must be an instance of MLP.Layer")
        super().__init__(*layers)

    class Layer(nn.Module):
        def __init__(
            self,
            in_size: int,
            out_size: Optional[int] = None,
            activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            dropout: float = 0.0,
            bias: bool = True,
        ):
            super().__init__()
            if activation is not None and not callable(activation):
                raise TypeError("activation must be callable: type={}".format(type(activation)))
            self.linear = nn.Linear(in_size, out_size, bias)
            self.activate = activation
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.linear(x)
            if self.activate is not None:
                h = self.activate(h)
            return self.dropout(h)


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # NOQA
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
