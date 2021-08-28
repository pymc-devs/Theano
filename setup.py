#!/usr/bin/env python
import sys

from setuptools import find_packages, setup

import versioneer


def read_file(filename):
    with open(filename, "rt") as buff:
        return buff.read()


NAME = "aesara"
MAINTAINER = "Aesara developers"
MAINTAINER_EMAIL = "aesara.devs@gmail.com"
DESCRIPTION = (
    "Optimizing compiler for evaluating mathematical expressions on CPUs and GPUs."
)
LONG_DESCRIPTION = read_file("DESCRIPTION.txt")
URL = "https://github.com/aesara-devs/aesara"
LICENSE = "BSD"
AUTHOR = "aesara-devs"
AUTHOR_EMAIL = "aesara.devs@gmail.com"
PLATFORMS = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
CLASSIFIERS = """\
Development Status :: 6 - Mature
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Code Generators
Topic :: Software Development :: Compilers
Topic :: Scientific/Engineering :: Mathematics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
"""
CLASSIFIERS = [_f for _f in CLASSIFIERS.split("\n") if _f]

install_requires = ["numpy>=1.17.0", "scipy>=0.14", "filelock"]

if sys.version_info[0:2] < (3, 7):
    install_requires += ["dataclasses"]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        platforms=PLATFORMS,
        packages=find_packages(exclude=["tests", "tests.*"]),
        install_requires=install_requires,
        package_data={
            "": [
                "*.txt",
                "*.rst",
                "*.cu",
                "*.cuh",
                "*.c",
                "*.sh",
                "*.pkl",
                "*.h",
                "*.cpp",
                "ChangeLog",
                "c_code/*",
            ],
            "aesara.misc": ["*.sh"],
            "aesara.d3viz": ["html/*", "css/*", "js/*"],
        },
        entry_points={
            "console_scripts": [
                "aesara-cache = bin.aesara_cache:main",
            ]
        },
        keywords=" ".join(
            [
                "aesara",
                "math",
                "numerical",
                "symbolic",
                "blas",
                "numpy",
                "gpu",
                "autodiff",
                "differentiation",
            ]
        ),
    )
