import sys
from pathlib import Path

wheels_dir = Path(sys.argv[1])

download_links = []
if wheels_dir.exists():
    wheels = wheels_dir.glob('*.whl')
    download_links = [
        '<li><a href={}>{}</a></li>'.format(
            wheel.as_posix(), wheel.name
        )
        for wheel in wheels
    ]

with open('index.html', 'w') as f:
    f.write('''<html>
<header>
    <title>A wonderfull doc...</title>
</header>

<body>

    Sphinx documentation is <a href="sphinx/sphinx-doc.html">here</a>

    Available wheels:
    <ul>
        {}
    </ul>
</body>

</html>'''.format('\n'.join(download_links)))
