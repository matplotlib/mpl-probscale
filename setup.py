# Setup script for the probscale package
#
# Usage: python setup.py install


import os
from setuptools import setup, find_packages


DESCRIPTION = "mpl-probscale: Probabily scales for matplotlib"
LONG_DESCRIPTION = DESCRIPTION
NAME = "probscale"
VERSION = "0.2.0"
AUTHOR = "Paul Hobson (Geosyntec Consultants)"
AUTHOR_EMAIL = "phobson@geosyntec.com"
URL = "https://github.com/phobson/mpl-probscale"
DOWNLOAD_URL = "https://github.com/phobson/mpl-probscale/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 2.7, 3.4 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]
INSTALL_REQUIRES = ['numpy', 'matplotlib']
PACKAGE_DATA = {
    'probscale.tests.baseline_images.test_viz': ['*png'],
    'probscale.tests.baseline_images.test_probscale': ['*png'],
}

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    #data_files=DATA_FILES,
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
)
