"""
setup.py for chroma-gui.

For reference see
https://packaging.python.org/guides/distributing-packages-using-setuptools/

"""
from pathlib import Path
from setuptools import setup, find_packages


HERE = Path(__file__).parent.absolute()
with (HERE / 'README.md').open('rt', encoding='utf-8') as fh:
    LONG_DESCRIPTION = fh.read().strip()

ABOUT_CHROMA_GUI: dict = {}
with (HERE / 'chroma_gui' / '__init__.py').open('rt') as fh:
    exec(fh.read(), ABOUT_CHROMA_GUI)


REQUIREMENTS: dict = {
    'core': [
        "matplotlib",
        "tfs-pandas",
        "pytimber",
        "pyqt5",
        "pandas",
        "numpy",
        "seaborn",
        "scipy",
    ],
    'test': [
        'pytest',
    ],
    'dev': [
        # 'requirement-for-development-purposes-only',
    ],
    'doc': [
        'sphinx',
        'acc-py-sphinx',
    ],
}


setup(
    name='chroma-gui',
    version=ABOUT_CHROMA_GUI['__version__'],
    author='Mäel Le Garrec',
    author_email='mael.le.garrec@cern.ch',
    description='Non-Linear Chromaticity GUI',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='',

    packages=find_packages(),
    python_requires='~=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    install_requires=REQUIREMENTS['core'],
    extras_require={
        **REQUIREMENTS,
        # The 'dev' extra is the union of 'test' and 'doc', with an option
        # to have explicit development dependencies listed.
        'dev': [req
                for extra in ['dev', 'test', 'doc']
                for req in REQUIREMENTS.get(extra, [])],
        # The 'all' extra is the union of all requirements.
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
    entry_points={
    'console_scripts': [
        'chroma-gui = chroma_gui.main:main',
        ],
    },
)
