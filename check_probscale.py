import sys
import matplotlib

matplotlib.use("agg")

from matplotlib.pyplot import style

style.use("classic")

from probscale import tests

status = tests.test(*sys.argv[1:])
sys.exit(status)
