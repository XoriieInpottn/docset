#!/usr/bin/env python3


import io
import struct

import numpy as np

from . import codec

INT64 = struct.Struct('<q')

DEFAULT_HEAD_SIZE = 4096
MIN_HEAD_SIZE = 512
DEFAULT_BLOCK_SIZE = 512


class DocSetWriter(object):

    def __init__(self, path, head_size=DEFAULT_HEAD_SIZE):
        self._path = path
        assert head_size >= MIN_HEAD_SIZE
        self._head_size = head_size

        self._closed = False
        self._fp = io.open(path, 'wb')

        self._index = []
        self._index_pos = None
        self._meta_doc = {}

        self._write_head()

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self._write_index()
            self._write_head()
            self._fp.close()
            self._closed = True

    def _write_head(self):
        basic_doc = {
            '__HDS__': self._head_size,  # head size
            '__CNT__': len(self._index),  # count of samples
            '__IDX__': self._index_pos  # index start
        }
        doc_data = codec.encode_doc({**basic_doc, **self._meta_doc})
        doc_data_size = len(doc_data)
        pad_size = self._head_size - (INT64.size + doc_data_size)
        if pad_size < 0:
            doc_data = codec.encode_doc(basic_doc)
            doc_data_size = len(doc_data)
            pad_size = self._head_size - (INT64.size + doc_data_size)
        self._fp.seek(0, io.SEEK_SET)
        self._fp.write(INT64.pack(doc_data_size))
        self._fp.write(doc_data)
        if pad_size > 0:
            self._fp.write(b'\0' * pad_size)
        self._fp.seek(0, io.SEEK_END)

    def write(self, doc):
        assert not self._closed
        pos = self._fp.tell()
        doc_data = codec.encode_doc(doc)
        self._fp.write(INT64.pack(len(doc_data)))
        self._fp.write(doc_data)
        self._index.append(pos)

    def _write_index(self):
        self._index_pos = self._fp.tell()
        for pos in self._index:
            self._fp.write(INT64.pack(pos))

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

    def __init__(self, path, block_size=DEFAULT_BLOCK_SIZE):
        self._path = path
        self._block_size = block_size

        self._closed = False
        self._fp = io.open(path, 'rb')

        doc_data_size = INT64.unpack(self._fp.read(INT64.size))[0]
        doc_data = self._fp.read(doc_data_size)
        doc = codec.decode_doc(doc_data)
        self._head_size = doc['__HDS__']
        self._index_count = doc['__CNT__']
        self._index_start = doc['__IDX__']
        self._index = np.full((self._index_count,), -1, dtype='<i8')
        self._meta_doc = {**doc}
        del self._meta_doc['__HDS__']
        del self._meta_doc['__CNT__']
        del self._meta_doc['__IDX__']

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self._fp.close()
            self._closed = True

    def read(self, i):
        pos = self._index[i]
        if pos < 0:
            i_left = (i // self._block_size) * self._block_size
            i_right = min(i_left + self._block_size, self._index_count)
            self._fp.seek(self._index_start + INT64.size * i_left, io.SEEK_SET)
            for j in range(i_left, i_right):
                self._index[j] = INT64.unpack(self._fp.read(INT64.size))[0]
            pos = self._index[i]

        self._fp.seek(pos, io.SEEK_SET)
        doc_data_size = INT64.unpack(self._fp.read(INT64.size))[0]
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


def DocSet(path: str, mode: str, *, head_size=4096, block_size=512):
    if mode == 'r':
        return DocSetReader(path, block_size=block_size)
    elif mode == 'w':
        return DocSetWriter(path, head_size=head_size)
    else:
        raise RuntimeError('"mode" should be one of {"r", "w"}.')
