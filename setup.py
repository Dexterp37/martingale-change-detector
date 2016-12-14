#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup

setup(name='martingale-change-detector',
    version='0.1.1',
    author='Roberto Agostino Vitillo, Alessio Placitelli',
    author_email='alessio.placitelli@gmail.com',
    description='A martingale approach to detect changes in Telemetry histograms',
    url='https://github.com/Dexterp37/martingale-change-detector',
    packages=['detector'],
    package_dir={'detector': 'detector'},
    install_requires=[
        'ujson',
        'numpy',
        'scipy'
    ],
    tests_require=['pytest']
)
