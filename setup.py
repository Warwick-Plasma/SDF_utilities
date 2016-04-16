import os
import sys
from distutils.sysconfig import get_python_lib
from distutils.core import setup, Extension
try:
    import numpy
    gotnumpy = True
except:
    gotnumpy = False


def get_numpy_dir():
    if gotnumpy:
        numpydir = os.path.dirname(numpy.__file__)
        incdir = os.path.join(numpydir, 'core', 'include')
        for r, d, fl in os.walk(incdir):
            for f in fl:
                if f == 'arrayobject.h':
                    return os.path.realpath(os.path.join(r, '..'))
        for r, d, fl in os.walk(numpydir):
            for f in fl:
                if f == 'arrayobject.h':
                    return os.path.realpath(os.path.join(r, '..'))
    sys.path.insert(0, get_python_lib(standard_lib=1))
    for path in sys.path:
        for r, d, fl in os.walk(path):
            for f in fl:
                if f == 'arrayobject.h':
                    return os.path.realpath(os.path.join(r, '..'))
    print('Unable to build python module. Numpy directory not found.')
    sys.exit(1)


sdfdir = os.path.join('..', 'C')

srcfiles = ['sdf_python.c']

incdirs = [get_numpy_dir()] + [os.path.join(sdfdir, 'src')]
libdirs = [os.path.join(sdfdir, 'lib')]

setup(name="sdf", version="1.0",
      ext_modules=[Extension("sdf", srcfiles, include_dirs=incdirs,
                   library_dirs=libdirs, libraries=['sdfc'])],
      py_modules=["sdf_legacy","sdf_helper"])
