#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="recall2predict",
    version="0.0.1",
    description="Recall2Predict AV2 motion forecasting training pipeline",
    author="",
    author_email="",
    url="https://github.com/abviv/recall2predict",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
