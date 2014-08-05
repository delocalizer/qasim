#!/usr/bin/env python3
import sys
import os, os.path
from distutils.core import setup
from distutils.extension import Extension
from distutils.dep_util import newer_group

try:
    from numpy import get_include as np_get_include
except ImportError:
    print("This extension requires numpy.")
    sys.exit(1)
try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Cython not present, compiling from distributed .c")
    HAVECYTHON = False
else:
    print("Cython present, building from .pyx")
    HAVECYTHON = True

sources_cython = [os.path.join('cython','qasim.pyx'),
                  os.path.join('cython','genreads.c'),]
sources_nocyth = [os.path.join('cython','qasim.c'),
                  os.path.join('cython','genreads.c'),]


args = sys.argv[1:]
# If command is sdist, bail if no Cython available, or if qasim.c is out
# of date - distribution must have up-to-date qasim.c file. 
if args.count('sdist'):
    if not HAVECYTHON:
        print("sdist for this package requires Cython.")
        sys.exit(3)
        
    if newer_group(sources_cython, os.path.join('cython','qasim.c')):
        print("qasim.c is out of date\nRun 'setup.py build_ext' "
              "before creating distribution")
        sys.exit(4)

# for other commands, start with new Cython'ed qasim.c file when possible.
else:
    if HAVECYTHON:
        try:
            os.remove(os.path.join('cython', 'qasim.c'))
        except FileNotFoundError:
            pass
        

name = 'Qasim'
cmdclass = {}
requires = ['numpy']
include_dirs = ['cython', np_get_include()]
ext_modules = []
packages = ['qasim']
package_dir = {'qasim':'qasim'}
version = '1.5.0'
author = 'Conrad Leonard'
author_email = 'c.leonard@imb.uq.edu.au'
platforms = ['linux']
ext_qasim = 'qasim.qasim'


if HAVECYTHON:
    e = Extension(ext_qasim, sources_cython, include_dirs)
    e.cython_directives = {"boundscheck":False}
    cmdclass.update({'build_ext': build_ext})
    ext_modules.append(e)
else:
    e = Extension(ext_qasim, sources_nocyth, include_dirs)
    ext_modules.append(e)


setup( 
    name = name, 
    cmdclass = cmdclass,
    requires = requires,
    ext_modules = ext_modules,
    packages = packages,
    package_dir = package_dir,
    version = version,
    author = author,
    author_email = author_email,
    platforms = platforms,
)
