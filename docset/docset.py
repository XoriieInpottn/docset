#!/usr/bin/env python3


import bisect
import io
import os
import struct
from typing import Sequence, List

import numpy as np
from bson import InvalidBSON

from . import codec
from . import docset_legacy

__all__ = [
    'DocSetWriter',
    'DocSetReader',
    'ConcatDocSet',
    'DocSet'
]

STRUCT_HEADER = struct.Struct('<8sQQQ')  # 32 B
UINT64 = struct.Struct('<Q')
TYPE_STR = b'DOCSET71'

NONE_VALUE = 0xFFFFFFFFFFFFFFFF
DEFAULT_BLOCK_SIZE = 512
DEFAULT_READ_BUFFER_SIZE = 1024 * 1024  # 1 MB
DEFAULT_WRITE_BUFFER_SIZE = 1024 * 1024 * 8  # 8 MB


class DocSetWriter(object):

    def __init__(self, path: str, buffer_size=DEFAULT_WRITE_BUFFER_SIZE):
        self._path = path
        self._index = []
        self._meta_doc = {}

        self._fp = io.open(path, 'wb', buffering=buffer_size)
        self._fp.write(STRUCT_HEADER.pack(TYPE_STR, 0, NONE_VALUE, NONE_VALUE))

    def __del__(self):
        self.close()

    def close(self):
        if self._fp is not None:
            count = len(self._index)
            index_start = self._write_index()
            meta_start = self._write_meta_doc()
            self._fp.seek(0, io.SEEK_SET)
            self._fp.write(STRUCT_HEADER.pack(TYPE_STR, count, index_start, meta_start))
            self._fp.close()
            self._fp = None

    def write(self, doc):
        assert self._fp is not None
        pos = self._fp.tell()
        doc_data = codec.encode_doc(doc)
        self._fp.write(UINT64.pack(len(doc_data)))
        self._fp.write(doc_data)
        self._index.append(pos)

    def _write_index(self):
        index_start = self._fp.tell()
        for pos in self._index:
            self._fp.write(UINT64.pack(pos))
        return index_start

    def _write_meta_doc(self):
        meta_start = self._fp.tell()
        doc_data = codec.encode_doc(self._meta_doc)
        self._fp.write(UINT64.pack(len(doc_data)))
        self._fp.write(doc_data)
        return meta_start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._index)

    @property
    def meta_doc(self):
        return self._meta_doc


class DocSetReader(object):

    def __init__(self,
                 path: str,
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 buffer_size: int = DEFAULT_READ_BUFFER_SIZE):
        self._path = path
        self._block_size = block_size
        self._buffer_size = buffer_size

        self._fp = io.open(self._path, 'rb', buffering=self._buffer_size)

        type_str, count, index_start, meta_start = STRUCT_HEADER.unpack(self._fp.read(STRUCT_HEADER.size))
        if type_str != TYPE_STR:
            raise RuntimeError('Invalid DocSet file.')
        if index_start == NONE_VALUE or meta_start == NONE_VALUE:
            raise RuntimeError('Incomplete DocSet.')
        self._index_start = index_start
        self._index_count = count
        self._fp.seek(meta_start, io.SEEK_SET)
        doc_data_size = UINT64.unpack(self._fp.read(UINT64.size))[0]
        doc_data = self._fp.read(doc_data_size)
        try:
            self._meta_doc = codec.decode_doc(doc_data)
        except InvalidBSON:
            raise RuntimeError('Invalid metadata.')

        self._index = np.full((self._index_count,), NONE_VALUE, dtype='<u8')

        self.pid = os.getpid()

    def __del__(self):
        self.close()

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def read(self, i):
        pid = os.getpid()
        if pid != self.pid:
            self._fp.close()
            self._fp = io.open(self._path, 'rb', buffering=self._buffer_size)
            self.pid = pid

        pos = self._index[i]
        if pos == NONE_VALUE:
            i_left = (i // self._block_size) * self._block_size
            i_right = min(i_left + self._block_size, self._index_count)
            self._fp.seek(self._index_start + UINT64.size * i_left, io.SEEK_SET)
            for j in range(i_left, i_right):
                self._index[j] = UINT64.unpack(self._fp.read(UINT64.size))[0]
            pos = self._index[i]

        self._fp.seek(pos, io.SEEK_SET)
        doc_data_size = UINT64.unpack(self._fp.read(UINT64.size))[0]
        doc_data = self._fp.read(doc_data_size)
        doc = codec.decode_doc(doc_data)
        return doc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._index)

    @property
    def meta_doc(self):
        return self._meta_doc

    def __getitem__(self, i):
        return self.read(i)


class ConcatDocSet(object):

    def __init__(self, ds_list: Sequence[DocSetReader]):
        self.ds_list: List[DocSetReader] = []
        self.cum_sizes = []

        assert len(ds_list) != 0
        cum_size = 0
        for ds in ds_list:
            self.ds_list.append(ds)
            cum_size = cum_size + len(ds)
            self.cum_sizes.append(cum_size)

    def __len__(self):
        return self.cum_sizes[-1]

    def read(self, i):
        if i < 0:
            if -i > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            i = len(self) + i
        ds_idx = bisect.bisect_right(self.cum_sizes, i)
        if ds_idx == 0:
            sample_idx = i
        else:
            sample_idx = i - self.cum_sizes[ds_idx - 1]
        return self.ds_list[ds_idx][sample_idx]

    def __getitem__(self, i):
        return self.read(i)


# noinspection PyPep8Naming
def DocSet(path: str, mode: str = 'r', *, block_size: int = None, buffer_size: int = None):
    if mode == 'r':
        if block_size is None:
            block_size = DEFAULT_BLOCK_SIZE
        if buffer_size is None:
            buffer_size = DEFAULT_READ_BUFFER_SIZE
        try:
            return DocSetReader(path, block_size=block_size, buffer_size=buffer_size)
        except RuntimeError:
            return docset_legacy.DocSetReader(path, block_size=block_size)
    elif mode == 'w':
        if buffer_size is None:
            buffer_size = DEFAULT_WRITE_BUFFER_SIZE
        return DocSetWriter(path, buffer_size=buffer_size)
    else:
        raise RuntimeError('"mode" should be one of {"r", "w"}.')
