import sys

from probscale import tests


status = tests.test(*sys.argv[1:])
sys.exit(status)
