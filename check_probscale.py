import sys
import matplotlib
matplotlib.use('agg')

import pytest
status = pytest.main(sys.argv[1:])
sys.exit(status)
