# Setup script for the probscale package
#
# Usage: python setup.py install

from setuptools import setup, find_packages


DESCRIPTION = "mpl-probscale: Probabily scales for matplotlib"
LONG_DESCRIPTION = DESCRIPTION
NAME = "probscale"
VERSION = "0.2.5"
AUTHOR = "Paul Hobson (Geosyntec Consultants)"
AUTHOR_EMAIL = "phobson@geosyntec.com"
URL = "https://github.com/matplotlib/mpl-probscale"
DOWNLOAD_URL = "https://github.com/matplotlib/mpl-probscale/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.5 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]
INSTALL_REQUIRES = ["numpy", "matplotlib"]
PACKAGE_DATA = {
    "probscale.tests.baseline_images.test_viz": ["*png"],
    "probscale.tests.baseline_images.test_probscale": ["*png"],
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
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
)
