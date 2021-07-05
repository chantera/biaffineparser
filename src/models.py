import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
        embeddings[0] = torch.zeros(embeddings[0])
        embeddings[1] = kwargs["pretrained_word_embeddings"]
        std = torch.std(embeddings[1])
        if std > 0:
            embeddings[1] /= std
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
    encoder.freeze_embedding(1)
    model = BiaffineParser(
        encoder,
        n_deprels=kwargs.get("n_deprels"),
        head_mlp_units=kwargs.get("head_mlp_units", 500),
        deprel_mlp_units=kwargs.get("deprel_mlp_units", 100),
        head_mlp_dropout=kwargs.get("head_mlp_dropout", dropout_ratio),
        deprel_mlp_dropout=kwargs.get("deprel_mlp_dropout", dropout_ratio),
    )
    return model


class BiaffineParser(nn.Module):
    def __init__(
        self,
        encoder: "Encoder",
        n_deprels: Optional[int] = None,
        head_mlp_units: Union[Sequence[int], int] = 500,
        deprel_mlp_units: Union[Sequence[int], int] = 100,
        head_mlp_dropout: float = 0.0,
        deprel_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(head_mlp_units, int):
            head_mlp_units = [head_mlp_units]
        if isinstance(deprel_mlp_units, int):
            deprel_mlp_units = [deprel_mlp_units]
        activation = partial(F.leaky_relu, negative_slope=0.1)

        def _create_mlp(in_size, units, dropout):
            return MLP(
                MLP.Layer(units[i - 1] if i > 0 else in_size, u, activation, dropout)
                for i, u in enumerate(units)
            )

        self.encoder = encoder
        in_size = self.encoder.out_size
        # NOTE: `in` and `out` are short for incoming (parent) and outgoing (child), respectively
        self.mlp_head_in = _create_mlp(in_size, head_mlp_units, head_mlp_dropout)
        self.mlp_head_out = _create_mlp(in_size, head_mlp_units, head_mlp_dropout)
        self.biaf_head = Biaffine(head_mlp_units[-1], head_mlp_units[-1], 1)
        self.mlp_deprel_in = None
        self.mlp_deprel_out = None
        self.biaf_deprel = None
        if n_deprels is not None:
            self.mlp_deprel_in = _create_mlp(in_size, deprel_mlp_units, deprel_mlp_dropout)
            self.mlp_deprel_out = _create_mlp(in_size, deprel_mlp_units, deprel_mlp_dropout)
            self.biaf_deprel = Biaffine(deprel_mlp_units[-1], deprel_mlp_units[-1], n_deprels)

    def forward(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        hs, lengths = self.encoder(*input_ids)  # => (b, n, d)
        hs_head_in = self.mlp_head_in(hs)
        hs_head_out = self.mlp_head_out(hs)
        logits_head = self.biaf_head(hs_head_out, hs_head_in).squeeze_(3)  # outgoing -> incoming
        mask = _mask_arc(lengths, mask_diag=not self.training)
        if mask is not None:
            logits_head.masked_fill_(mask.logical_not().to(logits_head.device), -float("inf"))
        logits_deprel = None
        if self.biaf_deprel is not None:
            assert self.mlp_deprel_in is not None and self.mlp_deprel_out
            hs_deprel_in = self.mlp_deprel_in(hs)
            hs_deprel_out = self.mlp_deprel_out(hs)
            logits_deprel = self.biaf_deprel(hs_deprel_out, hs_deprel_in)  # outgoing -> incoming
        return logits_head, logits_deprel, lengths  # (b, n, n), (b, n, n, n_deprels), (b,)

    def parse(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.decode(self.forward(input_ids))

    @torch.no_grad()
    def decode(
        self,
        logits_head: torch.Tensor,
        logits_deprel: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if lengths is None:
            lengths = torch.full((logits_head.size(0),), fill_value=logits_head.size(1))
        heads = _parse_graph(logits_head, lengths)
        deprels = _decode_deprels(logits_deprel, heads) if logits_deprel is not None else None
        return heads, deprels

    def compute_metrics(
        self,
        logits_head: torch.Tensor,
        logits_deprel: Optional[torch.Tensor],
        true_heads: Sequence[torch.Tensor],
        true_deprels: Optional[Sequence[torch.Tensor]],
    ) -> Dict[str, Any]:
        true_heads = nn.utils.rnn.pad_sequence(true_heads, batch_first=True, padding_value=-1)
        if true_deprels is not None:
            true_deprels = nn.utils.rnn.pad_sequence(
                true_deprels, batch_first=True, padding_value=-1
            )
        result = _compute_metrics(
            logits_head, true_heads, logits_deprel, true_deprels, ignore_index=-1
        )
        if result["deprel_loss"] is not None:
            result["loss"] = result["head_loss"] + result["deprel_loss"]
        else:
            result["loss"] = result["head_loss"]
        return result


@torch.no_grad()
def _mask_arc(lengths: torch.Tensor, mask_diag: bool = True) -> Optional[torch.Tensor]:
    b, n = lengths.numel(), lengths.max()
    if torch.all(lengths == n):
        if not mask_diag:
            return None
        mask = torch.ones(b, n, n)
    else:
        mask = torch.zeros(b, n, n)
        for i, length in enumerate(lengths):
            mask[i, :length, :length] = 1
    if mask_diag:
        mask.masked_fill_(torch.eye(n, dtype=torch.bool), 0)
    return mask


@torch.no_grad()
def _parse_graph(
    logits_head: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if mask is None:
        mask = _mask_arc(lengths, mask_diag=True)
    probs = (F.softmax(logits_head, dim=2) * mask.to(logits_head.device)).cpu().numpy()
    trees = torch.full((lengths.numel(), max(lengths)), -1)
    for i, length in enumerate(lengths):
        trees[i, :length] = torch.from_numpy(chuliu_edmonds_one_root(probs[i, :length, :length]))
    trees[:, 0] = -1
    return trees


@torch.no_grad()
def _decode_deprels(
    logits_deprel: torch.Tensor, trees: torch.Tensor, root: int = 0
) -> torch.Tensor:
    steps = torch.arange(trees.size(1))
    logits_deprel = [logits_deprel[i, steps, heads] for i, heads in enumerate(trees)]
    logits_deprel = torch.stack(logits_deprel, dim=0)
    logits_deprel[:, :, root] = -float("inf")
    deprels = logits_deprel.argmax(dim=2).detach().cpu()
    deprels.masked_fill_(trees == 0, root)
    return deprels


def _compute_metrics(
    logits_head: torch.Tensor,
    true_heads: torch.Tensor,
    logits_deprel: Optional[torch.Tensor] = None,
    true_deprels: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = -1,
    use_predicted_heads_for_deprels: bool = False,
) -> Dict[str, Any]:
    result = dict.fromkeys(["head_loss", "head_accuracy", "deprel_loss", "deprel_accuracy"])

    def metrics(y, t, ignore_index):
        loss = F.cross_entropy(y, t, ignore_index=ignore_index, reduction="sum")
        loss.div_((t != -1).sum())
        accuracy = categorical_accuracy(y, t, ignore_index)
        return loss, accuracy

    logits_head, true_heads = logits_head[:, 1:], true_heads[:, 1:]  # exclude root
    logits_head_flatten = logits_head.contiguous().view(-1, logits_head.size(-1))
    true_heads_flatten = true_heads.contiguous().view(-1)
    head_loss, head_accuracy = metrics(logits_head_flatten, true_heads_flatten, ignore_index)
    result.update(head_loss=head_loss, head_accuracy=head_accuracy)

    if logits_deprel is None:
        return result
    elif true_deprels is None:
        raise ValueError("'true_deprels' must be given to compute loss for deprels")

    if use_predicted_heads_for_deprels:
        heads = logits_head.argmax(dim=2)
    else:
        heads = true_heads.masked_fill(true_heads == -1, 0)

    logits_deprel, true_deprels = logits_deprel[:, 1:], true_deprels[:, 1:]  # exclude root
    gather_index = heads.view(*heads.size(), 1, 1).expand(-1, -1, -1, logits_deprel.size(-1))
    logits_deprel = torch.gather(logits_deprel, dim=2, index=gather_index)
    logits_deprel_flatten = logits_deprel.contiguous().view(-1, logits_deprel.size(-1))
    true_deprels_flatten = true_deprels.contiguous().view(-1)
    deprel_loss, deprel_accuracy = metrics(
        logits_deprel_flatten, true_deprels_flatten, ignore_index
    )
    result.update(deprel_loss=deprel_loss, deprel_accuracy=deprel_accuracy)

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
                emb = nn.Embedding.from_pretrained(item, freeze=False)
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

    def freeze_embedding(self, index: Optional[Union[int, Iterable[int]]] = None) -> None:
        if index is None:
            index = range(len(self.embeds))
        elif isinstance(index, int):
            index = [index]
        for i in index:
            self.embeds[i].weight.requires_grad = False

    def forward(self, *input_ids: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(input_ids) != len(self.embeds):
            raise ValueError(f"exact {len(self.embeds)} types of sequences must be given")
        lengths = torch.tensor([x.size(0) for x in input_ids[0]])
        xs = [emb(torch.cat(ids_each, dim=0)) for emb, ids_each in zip(self.embeds, input_ids)]
        if len(self._reduce_embs) > 1:
            xs += [torch.sum(torch.stack([xs.pop(i) for i in reversed(self._reduce_embs)]), dim=0)]
        seq = self.lstm_dropout(torch.cat(self.embedding_dropout(xs), dim=-1))  # (B * n, d)

        if torch.all(lengths == lengths[0]):
            hs, _ = self.bilstm(seq.view(len(lengths), lengths[0], -1))
        else:
            seq = torch.split(seq, tuple(lengths), dim=0)
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

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Drop embeddings with scaling.
        https://github.com/tdozat/Parser-v2/blob/304c638aa780a5591648ef27060cfa7e4bee2bd0/parser/neural/models/nn.py#L50  # noqa
        """
        if not self.training or self.p == 0.0:
            return list(xs)
        with torch.no_grad():
            masks = torch.rand((len(xs),) + xs[0].size()[:-1], device=xs[0].device) >= self.p
            scale = masks.size(0) / torch.clamp(masks.sum(dim=0, keepdims=True), min=1.0)
            masks = (masks * scale)[..., None]
        return [x * mask for x, mask in zip(xs, masks)]

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
            if id(self._flat_weights) != self._flat_weights_id:
                self.flatten_parameters()
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

    def flatten_parameters(self) -> None:
        super().flatten_parameters()
        self._flat_weights_id = id(self._flat_weights)


class MLP(nn.Sequential):
    def __init__(self, layers: Iterable["MLP.Layer"]):
        super().__init__(*layers)
        if not all(isinstance(layer, MLP.Layer) for layer in self):
            raise TypeError("each layer must be an instance of MLP.Layer")

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
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
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
