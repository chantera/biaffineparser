from collections import OrderedDict, UserDict
from os import PathLike
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")

TEnumerizer = TypeVar("TEnumerizer", bound="Enumerizer")


class Enumerizer(Collection[T]):
    def __init__(
        self,
        unknown: Optional[T] = None,
        mapping: Optional["IndexMapping[T]"] = None,
    ):
        self.mapping = IndexMapping() if mapping is None else mapping
        self.unknown_id = None if unknown is None else self.mapping[unknown]
        self.preprocess: Optional[Callable[[T], T]] = None

    def __call__(self, x: T) -> int:
        if callable(self.preprocess):
            x = self.preprocess(x)
        if self.mapping.increment:
            return self.mapping[x]
        index = self.mapping.get(x, self.unknown_id)
        if index is None:
            raise KeyError(x)
        return index

    def lookup(self, index: int) -> T:
        return self.mapping.lookup(index)

    def freeze(self):
        self.mapping.increment = False

    @classmethod
    def fromkeys(
        cls: Type[TEnumerizer],
        iterable: Iterable[T],
        unknown: Optional[T] = None,
    ) -> TEnumerizer:
        mapping = IndexMapping.fromkeys(iterable)
        if unknown is not None:
            mapping.increment = True
            _ = mapping[unknown]
        mapping.increment = False
        return cls(unknown, mapping)

    def __contains__(self, x) -> bool:
        return x in self.mapping

    def __iter__(self) -> Iterator[T]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, x) -> int:
        return self(x)


Vocab = Enumerizer[str]


class _UserDict(UserDict, MutableMapping[KT, VT]):
    pass


class IndexMapping(_UserDict[T, int]):
    def __init__(self):
        super().__init__()
        self.increment: bool = True
        self._index = -1
        self._idx2key = {}

    def lookup(self, index):
        return self._idx2key[index]

    @property
    def max(self):
        return self._index + 1

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def __missing__(self, key):
        if not self.increment:
            raise KeyError(key)
        self._index = idx = self._index + 1
        self.data[key] = idx
        self._idx2key[idx] = key
        return idx

    def __setitem__(self, key, item):
        if not isinstance(item, int):
            raise ValueError("item must be an int, but {} given".format(type(item)))
        if self._idx2key.get(item, key) != key:
            raise ValueError("`{}` has already been assigned to `{}`".format(item, key))
        if key in self.data:
            del self._idx2key[self.data[key]]
        self.data[key] = item
        self._idx2key[item] = key
        if item > self._index:
            self._index = item

    def __delitem__(self, key):
        del self._idx2key[self.data.pop(key)]

    def copy(self):
        idx2key = self._idx2key
        try:
            self._idx2key = {}
            c = super().copy()
        finally:
            self._idx2key = idx2key
        return c

    __marker = object()

    @classmethod
    def fromkeys(cls, iterable, value=__marker):
        if value is not cls.__marker:
            raise ValueError("value must not be specified")
        self = cls()
        idx = -1
        for idx, key in enumerate(iterable):
            self.data[key] = idx
            self._idx2key[idx] = key
        self._index = idx
        self.increment = False
        return self

    def pop(self, key, default=__marker):
        if key in self:
            result = self.data[key]
            del self.data[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default

    def popitem(self):
        result = self.data.popitem()
        del self._idx2key[result[1]]
        return result

    def clear(self):
        self.data.clear()
        self._idx2key.clear()
        self._index = -1

    def update(self, *args, **kwargs):
        d = dict()
        d.update(*args, **kwargs)
        for k, v in d.items():
            self[k] = v

    def setdefault(self, key, default=-1):
        if key in self.data:
            return self.data[key]
        self[key] = default
        return default


def load_embeddings(
    file: Union[str, bytes, PathLike],
    vocab_file: Optional[Union[str, bytes, PathLike]] = None,
    delimiter: str = " ",
) -> Dict[str, List[float]]:
    def _parse(file, vocab_file=None):
        if vocab_file:
            with open(file) as ef, open(vocab_file) as vf:
                for line1, line2 in zip(ef, vf):
                    token = line2.rstrip("\r\n")
                    vector = line1.rstrip("\r\n").split(delimiter)
                    yield (token, vector)
        else:
            with open(file) as f:
                if len(f.readline().rstrip("\r\n").split(delimiter)) > 2:
                    f.seek(0)
                for line in f:
                    entry = line.rstrip("\r\n").split(delimiter)
                    token, vector = entry[0], entry[1:]
                    yield (token, vector)

    embeddings = OrderedDict()
    for token, vector in _parse(file, vocab_file):
        if token in embeddings:
            raise ValueError(f"duplicate entry: {token!r}")
        embeddings[token] = [float(v) for v in vector]
    return embeddings
