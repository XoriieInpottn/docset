#!/usr/bin/env python3


from setuptools import setup

if __name__ == '__main__':
    with open('README.md') as file:
        long_description = file.read()
    setup(
        name='docset',
        packages=[
            'docset',
        ],
        entry_points={
            'console_scripts': [
                'docset = docset.entries:entry_docset'
            ]
        },
        version='0.5.1',
        keywords=('dataset', 'file format'),
        description='A dataset format/utilities used to store document objects based on BSON.',
        long_description_content_type='text/markdown',
        long_description=long_description,
        license='BSD-3-Clause License',
        author='xi',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/docset',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            'numpy',
            'pymongo'
        ]
    )
