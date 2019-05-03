import sys
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


#for the time being only with cython:
USE_CYTHON = True



extensions = Extension(
            name='precsum.precsum_impl',
            sources = ["src/precsum/precsum_impl.pyx"]
    )

if USE_CYTHON:
    extensions = cythonize(extensions, compiler_directives={'language_level' : sys.version_info[0]})

kwargs = {
      'name':'precsum',
      'version':'0.1.0',
      'description':'precise summation for floating point numbers',
      'author':'Egor Dranischnikow',
      'url':'https://github.com/realead/precsum',
      'packages':find_packages(where='src'),
      'package_dir':{"": "src"},
      'license': 'MIT',
      'ext_modules':  extensions,

       #ensure pxd-files:
      'package_data' : { 'precsum': ['*.pxd','*.pxi']},
      'include_package_data' : True,
      'zip_safe' : False  #needed because setuptools are used
}



setup(**kwargs)
