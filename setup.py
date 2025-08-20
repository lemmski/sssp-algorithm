"""
Setup script for building Cython-optimized FastSSSP extensions.

This script compiles the Cython modules for performance-critical
components of the FastSSSP algorithm.

Usage:
    python setup.py build_ext --inplace

Requirements:
    - Cython
    - NumPy
    - A C compiler (gcc, clang, or MSVC)
"""

from setuptools import setup, Extension
import numpy as np

# Cython extension modules
extensions = [
    Extension(
        "sssp_optimized",
        sources=["sssp_optimized.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],  # Optimization flags
        language="c",
    )
]

setup(
    name="fast_sssp_optimized",
    version="1.0.0",
    description="Cython-optimized FastSSSP algorithm",
    ext_modules=extensions,
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.0",
    ],
    python_requires=">=3.7",
)
