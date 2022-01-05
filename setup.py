# -*- coding: utf-8 -*-
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="wtil",
    version="1.0",
    description="imitation learning for wt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ppgame.com/rl/wtil",
    author="qinbo",
    author_email="qinbo@digisky.com",
    keywords="wtil",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "grpcio",
        "protobuf",
        "torch",
        "colorlog",
        "numpy",
        "tqdm",
        "sklearn",
        "tensorboard",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "black",
            "flake8",
            "nose",
            "grpcio-tools",
        ],
    },
)
