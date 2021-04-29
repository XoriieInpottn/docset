#!/usr/bin/env python3


import io
import os
import struct

import numpy as np
from bson import InvalidBSON

from . import codec

UINT64 = struct.Struct('<Q')

UNLOAD_VALUE = 0xFFFFFFFFFFFFFFFF
DEFAULT_HEAD_SIZE = 4096
MIN_HEAD_SIZE = 512
MAX_HEAD_SIZE = 16777216
DEFAULT_BLOCK_SIZE = 512


class DocSetWriter(object):

    def __init__(self, path: str, head_size: int = DEFAULT_HEAD_SIZE):
        self._path = path
        assert head_size >= MIN_HEAD_SIZE
        self._head_size = head_size

        self._fp = io.open(path, 'wb')

        self._index = []
        self._index_pos = None
        self._meta_doc = {}

        self._write_head()

    def __del__(self):
        self.close()

    def close(self):
        if self._fp is not None:
            self._write_index()
            self._write_head()
            self._fp.close()
            self._fp = None

    def _write_head(self):
        basic_doc = {
            '__HDS__': self._head_size,  # head size
            '__CNT__': len(self._index),  # count of samples
            '__IDX__': self._index_pos  # index start
        }
        doc_data = codec.encode_doc({**basic_doc, **self._meta_doc})
        doc_data_size = len(doc_data)
        pad_size = self._head_size - (UINT64.size + doc_data_size)
        if pad_size < 0:
            doc_data = codec.encode_doc(basic_doc)
            doc_data_size = len(doc_data)
            pad_size = self._head_size - (UINT64.size + doc_data_size)
        self._fp.seek(0, io.SEEK_SET)
        self._fp.write(UINT64.pack(doc_data_size))
        self._fp.write(doc_data)
        if pad_size > 0:
            self._fp.write(b'\0' * pad_size)
        self._fp.seek(0, io.SEEK_END)

    def write(self, doc):
        assert self._fp is not None
        pos = self._fp.tell()
        doc_data = codec.encode_doc(doc)
        self._fp.write(UINT64.pack(len(doc_data)))
        self._fp.write(doc_data)
        self._index.append(pos)

    def _write_index(self):
        self._index_pos = self._fp.tell()
        for pos in self._index:
            self._fp.write(UINT64.pack(pos))

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

    def __init__(self, path: str, block_size: int = DEFAULT_BLOCK_SIZE):
        self._path = path
        self._block_size = block_size

        self._fp = None

        with io.open(path, 'rb') as fp:
            doc_data_size = UINT64.unpack(fp.read(UINT64.size))[0]
            if doc_data_size > MAX_HEAD_SIZE or doc_data_size > os.path.getsize(path):
                raise RuntimeError('Invalid DocSet file.')
            doc_data = fp.read(doc_data_size)
            try:
                doc = codec.decode_doc(doc_data)
            except InvalidBSON:
                raise RuntimeError('Invalid DocSet file.')
            try:
                self._head_size = doc['__HDS__']
                self._index_count = doc['__CNT__']
                self._index_start = doc['__IDX__']
            except KeyError:
                raise RuntimeError('Corrupted meta document.')
            self._index = np.full((self._index_count,), UNLOAD_VALUE, dtype='<u8')
            self._meta_doc = {**doc}
            del self._meta_doc['__HDS__']
            del self._meta_doc['__CNT__']
            del self._meta_doc['__IDX__']

    def __del__(self):
        self.close()

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def read(self, i):
        if self._fp is None:
            self._fp = io.open(self._path, 'rb')

        pos = self._index[i]
        if pos == UNLOAD_VALUE:
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
