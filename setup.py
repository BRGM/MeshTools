# This is adapted from https://github.com/molpopgen/cmake_example/tree/subdir_and_python

import os
import re
import sys
from pathlib import Path
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from multiprocessing import cpu_count

sys.path.insert(0, Path('maintenance').as_posix())
from PackageInfo import PackageInfo

package = PackageInfo('MeshTools')

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        # depends option does not seem to work
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    
    def run(self):
        
        # check that cmake is present on the path and recent enough
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable]
        
        # this is where setuptools will build the module
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("")))
        setup_build_directory, setup_package_directory = os.path.split(extdir)
        assert setup_package_directory==self.package
        cmake_args+= ['-DCMAKE_INSTALL_PREFIX=' + setup_build_directory]
        # Link to CGAL directory
        def try_cgal_dir(variable):
            try:
                directory = os.environ[variable]
                if os.path.isdir(directory):
                    return directory
                print('WARNING: The %s environment variable references an non existing directory and will not be used!' % variable, file=sys.stderr)
            except KeyError:
                pass
        def find_cgal_dir():
            directory = try_cgal_dir('MESHTOOLS_WITH_CGAL_DIR')
            if directory is None:
                directory = try_cgal_dir('CGAL_DIR')
            return directory
        CGAL_DIR = find_cgal_dir()
        if CGAL_DIR:
            cmake_args+= ['-DCGAL_DIR=' + os.path.abspath(CGAL_DIR)]
            cmake_args+= ['-DBUILD_TESTING=0'] # not to build the CGAL test suite
            cmake_args+= ['-DCGAL_Boost_USE_STATIC_LIBS=1'] # to include the boost libraries and avoid linking problems
        else:
            print('WARNING: CGAL extensions will not be compiled!', file=sys.stderr)
            print('The CGAL_DIR or MESHTOOLS_WITH_CGAL_DIR environment variables can be used to point to the desired CGAL installation directory.', file=sys.stderr)
        build_args = ['--target', 'install'] # we will build the install target
        # config defaults to Debug on Windows if not passed to the build tool
        cfg = 'Debug' if self.debug else 'Release'
        build_args+= ['--config', cfg]
        # '--' will pass remaining options to the native build tool
        build_args+= ['--']
        
        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['-j%d' % cpu_count()]
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        # this is the build directory for compiled modules
        build_dir = os.path.abspath(self.build_temp) + '-cmake'
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        # cmake call to set up the native build
        cmake_cmd = ['cmake', ext.sourcedir] + cmake_args
        subprocess.check_call(cmake_cmd, cwd=build_dir, env=env)
        # cmake is then used to call the native build tool
        # just re-run cmake --build . --target install
        build_cmd = ['cmake', '--build', '.'] + build_args
        subprocess.check_call(build_cmd, cwd=build_dir)

setup(
    name=package.name,
    version=package.version,
    author='brgm',
    author_email='anr-charms@brgm.fr',
    description='A python library for managing mesh information and passing it to ComPASS.',
    long_description='',
    packages=['MeshTools', 'MeshTools.io', 'MeshTools.utils'],
    ext_package='MeshTools',
    ext_modules=[ CMakeExtension('_MeshTools') ], #name (first argument) must match the name of the exported pybind11 module
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
