import chainer
import chainer.functions as F
import numpy as np

from common import mst, nn


"""
NOTE: Weight Initialization comparison
  - A. This Implementation
    - URL: https://github.com/chantera/biaffineparser
    - Framework: Chainer (v5.2)
    - Initialization:
      - Embeddings (word): zero + pretrained (normalized with std)
      - Embeddings (postag): random_normal
      - BiLSTMs: `chainer.links.NStepBiLSTM` default
            (W: N(0,sqrt(1/w_in)), b: zero)
      - MLPs: W: N(0,sqrt(1/w_out)), b: zero
      - Biaffines: zero
  - B. Original
    - URL: https://github.com/tdozat/Parser-v1
      <tree:0739216129cd39d69997d28cbc4133b360ea3934>
    - Framework: TensorFlow (<v1.0, v0.8?)
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
      - See also: https://github.com/jcyk/Dynet-Biaffine-dependency-parser
    - Framework: MXNet (>=v1.3.0)
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

    def __init__(self, n_rels, encoder, encoder_dropout=0.0,
                 arc_mlp_units=500, rel_mlp_units=100,
                 arc_mlp_dropout=0.0, rel_mlp_dropout=0.0):
        super().__init__()
        if isinstance(arc_mlp_units, int):
            arc_mlp_units = [arc_mlp_units]
        if isinstance(rel_mlp_units, int):
            rel_mlp_units = [rel_mlp_units]
        with self.init_scope():
            def mlp_activate(x):
                # return F.maximum(0.1 * x, x)  # original
                return F.leaky_relu(x, slope=0.1)
            self.encoder = encoder
            h_dim = self.encoder.out_size
            init_mlp = chainer.initializers.HeNormal(
                scale=np.sqrt(0.5), fan_option='fan_out')
            self.mlp_arc_head = nn.MLP([nn.MLP.Layer(
                arc_mlp_units[i - 1] if i > 0 else h_dim, u, mlp_activate,
                arc_mlp_dropout, initialW=init_mlp,
                initial_bias=0.) for i, u in enumerate(arc_mlp_units)])
            self.mlp_arc_dep = self.mlp_arc_head.copy(mode='init')
            self.mlp_rel_head = nn.MLP([nn.MLP.Layer(
                rel_mlp_units[i - 1] if i > 0 else h_dim,  u, mlp_activate,
                rel_mlp_dropout, initialW=init_mlp,
                initial_bias=0.) for i, u in enumerate(rel_mlp_units)])
            self.mlp_rel_dep = self.mlp_rel_head.copy(mode='init')
            init_biaf = chainer.initializers.Zero()
            self.biaf_arc = nn.Biaffine(
                arc_mlp_units[-1], arc_mlp_units[-1], 1,
                nobias=(False, True, True),
                initialW=init_biaf, initial_bias=0.)
            self.biaf_rel = nn.Biaffine(
                rel_mlp_units[-1], rel_mlp_units[-1], n_rels,
                nobias=(False, False, False),
                initialW=init_biaf, initial_bias=0.)
        self.encoder_dropout = encoder_dropout
        self._results = {}

    def forward(self, words, pretrained_words, postags, *args):
        self._results.clear()
        # [n; B], [n; B], [n; B] => [(n, d); B]
        self._hs = self.encoder(words, pretrained_words, postags)
        self._lengths = [hs_seq.shape[0] for hs_seq in self._hs]
        # => (B, n_max, d)
        hs = nn.dropout(F.pad_sequence(self._hs), self.encoder_dropout)
        hs_arc_h = self.mlp_arc_head(hs, n_batch_axes=2)
        hs_arc_d = self.mlp_arc_dep(hs, n_batch_axes=2)
        hs_rel_h = self.mlp_rel_head(hs, n_batch_axes=2)
        hs_rel_d = self.mlp_rel_dep(hs, n_batch_axes=2)
        # => (B, n_max, n_max), (B, n_max, n_max, n_rels)
        self._logits_arc = F.squeeze(self.biaf_arc(hs_arc_d, hs_arc_h), axis=3)
        self._logits_rel = self.biaf_rel(hs_rel_d, hs_rel_h)
        return self._logits_arc, self._logits_rel

    def parse(self, words, pretrained_words, postags, use_cache=True):
        with chainer.no_backprop_mode():
            if len(self._results) == 0 or not use_cache:
                self.forward(words, pretrained_words, postags)
            return _parse(self._logits_arc, self._logits_rel, self._lengths)

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
        return arc_accuracy.data, rel_accuracy.data


def _parse(logits_arc, logits_rel, lengths):
    xp = chainer.cuda.get_array_module(logits_arc)
    mask = xp.full(logits_arc.shape, -float('inf'))
    for i, length in enumerate(lengths):
        mask[i, :, :length] = 0.0
    arc_probs = F.softmax(logits_arc.data + mask, axis=2).data
    arc_probs = chainer.cuda.to_cpu(arc_probs)
    rel_probs = F.softmax(logits_rel, axis=3).data
    rel_probs = chainer.cuda.to_cpu(rel_probs)
    parsed = [mst.mst(arc_prob[:length, :length], rel_prob[:length, :length])
              for arc_prob, rel_prob, length
              in zip(arc_probs, rel_probs, lengths)]
    return parsed


def _compute_metrics(parsed, gold_batch, lengths,
                     use_predicted_arcs_for_rels=True):
    logits_arc, logits_rel = parsed
    true_arcs, true_rels = zip(*gold_batch)

    # exclude attachment from the root
    logits_arc, logits_rel = logits_arc[:, 1:], logits_rel[:, 1:]
    true_arcs = F.pad_sequence(true_arcs, padding=-1)[:, 1:]
    true_rels = F.pad_sequence(true_rels, padding=-1)[:, 1:]
    lengths = np.array(lengths, dtype=np.int32) - 1
    xp = chainer.cuda.get_array_module(logits_arc)
    if xp is not np:
        true_arcs.to_gpu()
        true_rels.to_gpu()

    b, n_deps, n_heads = logits_arc.shape
    logits_arc_flatten = F.reshape(logits_arc, (b * n_deps, n_heads))
    true_arcs_flatten = F.reshape(true_arcs, (b * n_deps,))
    arc_loss = F.softmax_cross_entropy(
        logits_arc_flatten, true_arcs_flatten, ignore_label=-1)
    arc_accuracy = F.accuracy(
        logits_arc_flatten, true_arcs_flatten, ignore_label=-1)
    arc_accuracy.to_cpu()

    if use_predicted_arcs_for_rels:
        parsed_arcs = xp.argmax(logits_arc.data, axis=2)
    else:
        parsed_arcs = true_arcs.data
    logits_rel = [logits[np.arange(length), arcs[:length]]
                  for logits, arcs, length
                  in zip(logits_rel, parsed_arcs, lengths)]
    logits_rel = F.pad_sequence(logits_rel)
    b, n_deps, n_rels = logits_rel.shape
    logits_rel_flatten = F.reshape(logits_rel, (b * n_deps, n_rels))
    true_rels_flatten = F.reshape(true_rels, (b * n_deps,))
    rel_loss = F.softmax_cross_entropy(
        logits_rel_flatten, true_rels_flatten, ignore_label=-1)
    rel_accuracy = F.accuracy(
        logits_rel_flatten, true_rels_flatten, ignore_label=-1)
    rel_accuracy.to_cpu()

    return {'arc_loss': arc_loss, 'arc_accuracy': arc_accuracy,
            'rel_loss': rel_loss, 'rel_accuracy': rel_accuracy}


class Encoder(chainer.Chain):

    def __init__(self,
                 word_embeddings,
                 pretrained_word_embeddings=None,
                 postag_embeddings=None,
                 n_lstm_layers=3,
                 lstm_hidden_size=None,
                 embeddings_dropout=0.0,
                 lstm_dropout=0.0):
        super().__init__()
        with self.init_scope():
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
            embed_list = []
            for weights, fixed in embeddings:
                s = weights.shape
                embed_list.append(nn.EmbedID(s[0], s[1], weights, None, fixed))
            self.embeds = nn.EmbedList(embed_list, dropout=0.0, merge=False)
            if lstm_hidden_size is None:
                lstm_hidden_size = lstm_in_size
            # NOTE(chantera): The original implementation uses BiLSTM
            # with variational dropout for inputs and hidden states.
            # The same dropout is applied by the following code:
            # ---
            # self.bilstm = nn.NStepBiLSTM(
            #     n_lstm_layers, lstm_in_size, lstm_hidden_size, lstm_dropout,
            #     recurrent_dropout=lstm_dropout, use_variational_dropout=True)
            self.bilstm = chainer.links.NStepBiLSTM(
                n_lstm_layers, lstm_in_size, lstm_hidden_size, lstm_dropout)
        self.embeddings_dropout = embeddings_dropout
        self.lstm_dropout = lstm_dropout
        self._hidden_size = lstm_hidden_size

    def forward(self, *xs):
        # [(n, d_word); B], [(n, d_word); B], [(n, d_pos); B]
        # => [(n, d_word + d_pos); B]
        rs = self._concat_embeds(self.embeds(*xs))
        rs = [nn.dropout(rs_seq, self.lstm_dropout) for rs_seq in rs]
        return self.bilstm(hx=None, cx=None, xs=rs)[-1]

    def _concat_embeds(self, embed_outputs):
        rs_postags = embed_outputs.pop() if self._use_postag else None
        rs_words_pretrained = embed_outputs.pop() \
            if self._use_pretrained_word else None
        rs_words = embed_outputs.pop()
        if rs_words_pretrained is not None:
            rs_words = [rs_word_seq + rs_pre_seq for rs_word_seq, rs_pre_seq
                        in zip(rs_words, rs_words_pretrained)]
        rs_words, rs_postags = \
            _embed_dropout(rs_words, rs_postags,
                           self.embeddings_dropout, self.embeddings_dropout)
        if rs_postags:
            rs = [F.concat((rs_word_seq, rs_postag_seq))
                  for rs_word_seq, rs_postag_seq in zip(rs_words, rs_postags)]
        else:
            rs = rs_words
        return rs

    @property
    def out_size(self):
        return self._hidden_size * 2


def _embed_dropout(rs_words, rs_postags=None,
                   word_dropout=0.0, postag_dropout=0.0):
    """
    Drop words and tags with scaling to compensate the dropped one.
    https://github.com/tdozat/Parser-v1/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L58  # NOQA
    """
    if not chainer.config.train:
        return rs_words, rs_postags
    xp = chainer.cuda.get_array_module(rs_words[0])
    mask_shape = (len(rs_words), max(seq.shape[0] for seq in rs_words), 1)
    word_mask = xp.float32(1. - word_dropout) \
        * (xp.random.rand(*mask_shape) >= word_dropout)
    if rs_postags is not None:
        postag_mask = xp.float32(1. - postag_dropout) \
            * (xp.random.rand(*mask_shape) >= postag_dropout)
        word_embed_size = rs_words[0].shape[0]
        postag_embed_size = rs_postags[0].shape[0]
        embed_size = word_embed_size + postag_embed_size
        dropped_sizes = word_mask * word_embed_size \
            + postag_mask * postag_embed_size
        if word_embed_size == postag_embed_size:
            embed_size += word_embed_size
            dropped_sizes += word_mask * postag_mask * word_embed_size
        scale = embed_size / (dropped_sizes + 1e-12)
        word_mask *= scale
        postag_mask *= scale
    ys_words = [rs_word_seq * word_mask[i, :rs_word_seq.shape[0]]
                for i, rs_word_seq in enumerate(rs_words)]
    if rs_postags is not None:
        ys_postags = [rs_postag_seq * postag_mask[i, :rs_postag_seq.shape[0]]
                      for i, rs_postag_seq in enumerate(rs_postags)]
    else:
        ys_postags = None
    return ys_words, ys_postags
