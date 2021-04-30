#!/usr/bin/env python3


import io
import struct

import numpy as np
from bson import InvalidBSON
from . import docset_legacy

from . import codec

STRUCT_HEADER = struct.Struct('<8sQQQ')
TYPE_STR = b'DOCSET\0\0'
UINT64 = struct.Struct('<Q')

UNLOAD_VALUE = 0xFFFFFFFFFFFFFFFF
DEFAULT_BLOCK_SIZE = 512


class DocSetWriter(object):

    def __init__(self, path: str):
        self._path = path
        self._index = []
        self._meta_doc = {}

        self._fp = io.open(path, 'wb')
        self._fp.write(STRUCT_HEADER.pack(TYPE_STR, 0, UNLOAD_VALUE, UNLOAD_VALUE))

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

    def __init__(self, path: str, block_size: int = DEFAULT_BLOCK_SIZE):
        self._path = path
        self._block_size = block_size

        self._fp = None

        with io.open(path, 'rb') as fp:
            type_str, count, index_start, meta_start = STRUCT_HEADER.unpack(fp.read(STRUCT_HEADER.size))
            if type_str != TYPE_STR:
                raise RuntimeError('Invalid DocSet file.')
            if index_start == UNLOAD_VALUE or meta_start == UNLOAD_VALUE:
                raise RuntimeError('Incomplete DocSet.')
            self._index_start = index_start
            self._index_count = count
            fp.seek(meta_start, io.SEEK_SET)
            doc_data_size = UINT64.unpack(fp.read(UINT64.size))[0]
            doc_data = fp.read(doc_data_size)
            try:
                self._meta_doc = codec.decode_doc(doc_data)
            except InvalidBSON:
                raise RuntimeError('Invalid metadata.')

            self._index = np.full((self._index_count,), UNLOAD_VALUE, dtype='<u8')

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


def DocSet(path: str, mode: str, *, block_size=512):
    if mode == 'r':
        try:
            return DocSetReader(path, block_size=block_size)
        except RuntimeError:
            return docset_legacy.DocSetReader(path, block_size=block_size)
    elif mode == 'w':
        return DocSetWriter(path)
    else:
        raise RuntimeError('"mode" should be one of {"r", "w"}.')
