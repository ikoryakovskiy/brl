from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#ext_modules = Extension("fastloop",
#              ["conditional.pyx"],
#              extra_compile_args = ["-ffast-math"])

ext_modules = Extension("fastloop",
              ["conditional.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"])
#              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#              extra_link_args=['-fopenmp'])

setup(
  name = "fastloop",
  cmdclass = {"build_ext": build_ext},
  ext_modules = [ext_modules])