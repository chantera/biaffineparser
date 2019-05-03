from collections import Counter

import numpy as np
from teras.dataset.loader import TextLoader
from teras.io.reader import ConllReader
from teras.preprocessing import text


class DataLoader(TextLoader):

    def __init__(self,
                 word_embed_size=100,
                 postag_embed_size=100,
                 word_embed_file=None,
                 word_preprocess=text.lower,
                 word_unknown="<UNK>",
                 input_file=None,
                 min_frequency=2):
        super().__init__(reader=ConllReader())
        words = []
        if input_file is not None:
            self._fix_word = True
            counter = Counter([word_preprocess(token['form'])
                               for sentence in self._reader.read(input_file)
                               for token in sentence])
            for word, count in counter.most_common():
                if count < min_frequency:
                    break
                words.append(word)
        else:
            self._fix_word = False
        word_vocab = text.EmbeddingVocab.from_words(
            words, unknown=word_unknown, dim=word_embed_size,
            initializer=(text.EmbeddingVocab.random_normal
                         if word_embed_file is None else np.zeros),
            serialize_embeddings=True)
        pretrained_word_vocab = text.EmbeddingVocab(
            unknown=word_unknown, file=word_embed_file, dim=word_embed_size,
            initializer=np.zeros, serialize_embeddings=True)
        pretrained_word_vocab.add(word_preprocess("<ROOT>"))
        postag_vocab = text.EmbeddingVocab(
            unknown=word_unknown, dim=postag_embed_size,
            initializer=text.EmbeddingVocab.random_normal,
            serialize_embeddings=True)
        self.add_processor(
            'word', word_vocab, preprocess=word_preprocess)
        self.add_processor(
            'pre', pretrained_word_vocab, preprocess=word_preprocess)
        self.add_processor('pos', postag_vocab, preprocess=False)
        self.rel_map = text.Dict()

    def map(self, item):
        words, postags, heads, rels = zip(*[
            (token['form'], token['postag'], token['head'], token['deprel'])
            for token in item])
        word_ids = self.map_attr('word', words,
                                 False if self._fix_word else self.train)
        pre_ids = self.map_attr('pre', words, False)
        postag_ids = self.map_attr('pos', postags, self.train)
        heads = np.array(heads, dtype=np.int32)
        rel_ids = np.array(
            [self.rel_map.add(rel) for rel in rels], dtype=np.int32)
        sentence = None if self.train else item
        sample = (word_ids, pre_ids, postag_ids, sentence, (heads, rel_ids))
        return sample
