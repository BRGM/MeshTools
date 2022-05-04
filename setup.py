# dirty workaround around sysconfig.get_platform bug on MacOSX
import os
import sysconfig

platform = sysconfig.get_platform()
if platform.startswith("macosx"):
    assert all(platform.split("-")), "Cannot used platform information!"
    os.environ["_PYTHON_HOST_PLATFORM"] = platform


from skbuild import setup

# packages must be listed in setup.py to be intercepted by scikit-build
# cf. https://scikit-build.readthedocs.io/en/latest/usage.html#setuptools-options
setup(packages=["MeshTools", "MeshTools.io"])
