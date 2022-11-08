#!/usr/bin/env python3


import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from .docset import DocSet, ConcatDocSet

try:
    import cv2 as cv
except ImportError:
    cv = None


def view_ds(args):
    if len(args.input) == 0:
        print(f'At least 1 file should be given.', file=sys.stderr)
        return 1

    path = args.input[0]
    if not os.path.exists(path):
        print(f'File {path} does not exists.', file=sys.stderr)
        return 1

    try:
        with DocSet(path, 'r') as f:
            count = len(f)
            size = os.path.getsize(path)
            size_per_sample = int(size / count + 0.5) if count > 0 else 0
            print(path)
            print(f'Count: {count}, Size: {_format_size(size)}, Avg: {_format_size(size_per_sample)}/sample')
            print()

            meta_doc = f.meta_doc
            if meta_doc:
                for name, value in f.meta_doc.items():
                    print(f'{name}: {str(value)}')
                print()

            for i in range(min(2, count)):
                print(f'Sample {i}')
                _print_doc(f[i])

            if count > 2:
                i = count - 1
                print('...')
                print(f'Sample {i}')
                _print_doc(f[i])
    except RuntimeError as e:
        print(e, file=sys.stderr)
    return 0


def _format_size(size):
    _POWER_LABELS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    n = 0
    while size >= 1024:
        if n + 1 >= len(_POWER_LABELS):
            break
        size /= 1024
        n += 1
    if n == 0:
        return f'{size} {_POWER_LABELS[n]}'
    return f'{size:.01f} {_POWER_LABELS[n]}'


def _print_doc(doc):
    for name, value in doc.items():
        value_str = None
        if isinstance(value, str):
            value_str = f'"{value}"'
        elif isinstance(value, np.ndarray):
            value_str = f'ndarray(dtype={value.dtype}, shape={value.shape})'
        elif isinstance(value, bytes):
            if cv is not None:
                if value.startswith(b'\xff\xd8') and len(value) >= 10 and value[6:10] == b'JFIF':
                    try:
                        image = cv.imdecode(np.frombuffer(value, np.byte), cv.IMREAD_UNCHANGED)
                    except cv.error:
                        image = None
                    if image is not None:
                        value_str = f'jpeg_image(size={image.shape})'
                elif value.startswith(b'\x89\x50\x4e\x47'):
                    try:
                        image = cv.imdecode(np.frombuffer(value, np.byte), cv.IMREAD_UNCHANGED)
                    except cv.error:
                        image = None
                    if image is not None:
                        value_str = f'png_image(size={image.shape})'
            if value_str is None:
                size = _format_size(len(value))
                value_str = f'binary(size={size})'
        else:
            value_str = value
        print(f'    "{name}": {value_str}')


def merge_ds(args):
    if len(args.input) < 3:
        print('At least 3 ds files should be given. The last one indicates the output file.', file=sys.stderr)
        return 1
    if os.path.exists(args.input[-1]):
        print(f'{args.input[-1]} exists. Please remove it if you really want to do this.', file=sys.stderr)
        return 2
    for path in args.input[:-1]:
        if not os.path.exists(path):
            print(f'File "{path}" does not exist.', file=sys.stderr)
            return 3

    ds_list = [DocSet(path) for path in args.input[:-1]]
    src_ds = ConcatDocSet(ds_list)
    with DocSet(args.input[-1], 'w') as dst_ds:
        for doc in tqdm(src_ds, desc='Processing', leave=False, ncols=96):
            dst_ds.write(doc)
    for ds in ds_list:
        ds.close()
    return 0


_DISPATCH = {
    'view': view_ds,
    'merge': merge_ds
}


def entry_docset():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    cmd = args.input[0]
    if cmd in _DISPATCH:
        args.input = args.input[1:]
        return _DISPATCH[cmd](args)
    elif os.path.exists(cmd):
        return view_ds(args)
    else:
        print(f'Unknown command {cmd}.', file=sys.stderr)
        return -1


if __name__ == '__main__':
    raise SystemExit(entry_docset())
