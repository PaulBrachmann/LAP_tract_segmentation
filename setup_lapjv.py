import os
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "linear_assignment",
        [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "linear_assignment.pyx"
            )
        ],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
