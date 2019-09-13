import sys
from pathlib import Path

wheels_dir = Path(sys.argv[1])

download_links = []
if wheels_dir.exists():
    wheels = wheels_dir.glob('*.whl')
    download_links = [
        '  * `{} <{}>`_'.format(
            wheel.name, wheel.as_posix()
        )
        for wheel in wheels
    ]

with open('available_wheels.rst', 'w') as f:
    f.write('''
MeshTools wheels
================

Currently available wheels are:
{}

'''.format('\n'.join(download_links)))
