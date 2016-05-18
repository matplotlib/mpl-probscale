from pkg_resources import resource_filename

import pytest

import probscale

def test(*args):
    options = [resource_filename('probscale', 'tests')]
    options.extend(list(args))
    return pytest.main(options)