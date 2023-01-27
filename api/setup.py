import os

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.0.1"

# examples/tutorials
EXTRAS_REQUIRE = [
]

setup(
    name="pyciemss_api",
    version=VERSION,
    description="API for pyciemss",
    packages=find_packages(include=["pyciemss_api"]),
    author="Matt Printz",
    install_requires=[
        "fastapi",
        "uvicorn",
        "pyciemss",
        "causal_pyro",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
            "black",
            "flake8",
            "isort",
            "sphinx",
            "sphinx_rtd_theme",
            "myst_parser",
            "nbsphinx",
            "httpx",
        ],
    },
    python_requires=">=3.8",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10.7",
    ],
    # yapf
)
