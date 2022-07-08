# -*- ecoding: utf-8 -*-
# @ModuleName: setup
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:56 PM
from setuptools import setup, find_packages
from distutils.core import setup

with open("Readme.md", "r") as f:
    long_description = f.read()

setup(
    name='rec_pangu',
    version='0.0.7',
    description='Some Rank/Multi-task model implemented by Pytorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='wk',
    author_email='306178200@qq.com',
    url='https://github.com/HaSai666/rec_pangu',
    install_requires=['numpy>=1.19.0', 'torch>=1.7.0', 'pandas>=1.0.5', 'tqdm', 'scikit_learn>=0.23.2'],
    packages=find_packages(),
    platforms=["all"],
    classifiers=[
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=['rank', 'multi task', 'deep learning', 'pytorch', 'recsys', 'recommendation'],
)