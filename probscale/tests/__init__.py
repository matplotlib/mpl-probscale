from pkg_resources import resource_filename

import probscale


def test(*args):
    try:
        import pytest
    except ImportError:
        raise ImportError("`pytest` is required to run the test suite")

    options = [resource_filename("probscale", "tests")]
    options.extend(list(args))
    return pytest.main(options)
