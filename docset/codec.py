#!/usr/bin/env python3

import io

import numpy as np
from bson import BSON
from bson.binary import Binary, USER_DEFINED_SUBTYPE
from bson.codec_options import TypeCodec, CodecOptions, TypeRegistry

__all__ = [
    'encode_ndarray',
    'decode_ndarray',
    'encode_doc',
    'decode_doc'
]


def encode_ndarray(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    buf.write(a.dtype.str.encode())
    buf.write(str(a.shape).encode())
    buf.write(a.tobytes('C'))
    return buf.getvalue()


def decode_ndarray(data: bytes) -> np.ndarray:
    dtype_end = data.find(b'(')
    shape_start = dtype_end + 1
    shape_end = data.find(b')', shape_start)
    dtype = data[:dtype_end]
    shape = tuple(int(size) for size in data[shape_start:shape_end].split(b',') if size)
    buffer = data[shape_end + 1:]
    a = np.ndarray(dtype=dtype, shape=shape, buffer=buffer)
    return np.array(a)


class NumpyCodec(TypeCodec):
    sub_type = USER_DEFINED_SUBTYPE + 1
    python_type = np.ndarray
    bson_type = Binary

    def transform_python(self, a: np.ndarray):
        data = encode_ndarray(a)
        return Binary(data, NumpyCodec.sub_type)

    def transform_bson(self, data: Binary):
        if data.subtype == NumpyCodec.sub_type:
            return decode_ndarray(data)
        return data


CODEC_OPTIONS = CodecOptions(type_registry=TypeRegistry([NumpyCodec()]))


def encode_doc(doc) -> bytes:
    return BSON.encode(doc, codec_options=CODEC_OPTIONS)


def decode_doc(data: bytes):
    return BSON(data).decode(codec_options=CODEC_OPTIONS)
