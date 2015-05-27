import os, sys
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
    for r,d,fl in os.walk(incdir):
      for f in fl:
        if f == 'arrayobject.h':
          return os.path.realpath(os.path.join(r,'..'))
    for r,d,fl in os.walk(numpydir):
      for f in fl:
        if f == 'arrayobject.h':
          return os.path.realpath(os.path.join(r,'..'))
  sys.path.insert(0,get_python_lib(standard_lib=1))
  for path in sys.path:
    for r,d,fl in os.walk(path):
      for f in fl:
        if f == 'arrayobject.h':
          return os.path.realpath(os.path.join(r,'..'))
  print('Unable to build python module. Numpy directory not found.')
  sys.exit(1)


sdfdir = os.path.join('..','C','src')
sdffiles = ['sdf_control.c', 'sdf_derived.c', 'sdf_extension_util.c',
            'sdf_helper.c', 'sdf_input.c', 'sdf_input_cartesian.c', 'sdf_input_point.c',
            'sdf_input_station.c', 'sdf_util.c', 'stack_allocator.c']
sdffiles = [os.path.join(sdfdir,x) for x in sdffiles]

srcfiles = ['sdf_python.c'] + sdffiles

incdirs = [get_numpy_dir()] + [sdfdir]

setup(name="sdf", version="1.0",
      ext_modules=[Extension("sdf", srcfiles, include_dirs=incdirs)],
      py_modules=["sdf_legacy"])
