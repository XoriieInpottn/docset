#!/usr/bin/env python3


import argparse
import os
import sys

import numpy as np

from .docset import DocSet


def entry_docset():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1)
    args = parser.parse_args()

    path = args.input[0]
    if not os.path.exists(path):
        print(f'File {path} does not exists.', file=sys.stderr)
        return 1

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
        if isinstance(value, str):
            value = f'"{value}"'
        elif isinstance(value, np.ndarray):
            value = f'ndarray(dtype={value.dtype}, shape={value.shape})'
        elif isinstance(value, bytes):
            size = _format_size(len(value))
            value = f'binary(size={size})'
        print(f'    "{name}": {value}')


if __name__ == '__main__':
    raise SystemExit(entry_docset())
