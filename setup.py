# Setup script for the probscale package
#
# Usage: python setup.py install
import re
from setuptools import setup, find_packages


def search(substr: str, content: str):
    found = re.search(substr, content)
    if found:
        return found.group(1)
    return ""


with open("cloudside/__init__.py", encoding="utf8") as f:
    content = f.read()
    version = search(r'__version__ = "(.*?)"', content)
    author = search(r'__author__ = "(.*?)"', content)
    author_email = search(r'__email__ = "(.*?)"', content)


DESCRIPTION = "mpl-probscale: Probabily scales for matplotlib"
LONG_DESCRIPTION = DESCRIPTION
NAME = "probscale"
URL = "https://github.com/matplotlib/mpl-probscale"
DOWNLOAD_URL = "https://github.com/matplotlib/mpl-probscale/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.8 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
INSTALL_REQUIRES = ["numpy", "matplotlib"]
PACKAGE_DATA = {
    "probscale.tests.baseline_images.test_viz": ["*png"],
    "probscale.tests.baseline_images.test_probscale": ["*png"],
}

setup(
    name=NAME,
    version=version,
    author=author,
    author_email=author_email,
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
