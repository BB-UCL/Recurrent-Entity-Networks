try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'REN',
    'author': 'Raza Habib',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'raza.habib@cs.ucl.ac.uk',
    'version': '0.1',
    'install_requires': ['nose','theano','numpy'],
    'packages': ['REN'],
    'scripts': [],
    'name': 'REN'
}

setup(**config)
