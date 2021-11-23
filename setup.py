# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fDAL",
    version="0.0.1",
    author="David Acuna",
    author_email="dacunamarrer@nvidia.com",
    description="f-Domain Adversarial Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nv-tlabs/fdal",
    packages=['fDAL'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)