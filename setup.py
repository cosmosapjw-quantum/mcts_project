# setup.py
import os
import sys
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Add your own .cpp files below, including gomoku.cpp / attack_defense.cpp
sources = [
    "src/python_wrapper.cpp",
    "src/mcts.cpp",
    "src/node.cpp",
    "src/mcts_config.cpp",
    "src/nn_interface.cpp",
    # your original modules:
    "src/gomoku.cpp",
    "src/attack_defense.cpp",
]

include_dirs = [
    "include",
    pybind11.get_include()
]

ext = Extension(
    "mcts_py",  # The name of the Python module
    sources=sources,
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=["-std=c++17"]
)

setup(
    name="mcts_project",
    version="0.1.0",
    description="MCTS with Gomoku & Attack-Defense, plus Python self-play/arena",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["pybind11"]
)
