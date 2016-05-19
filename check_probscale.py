import sys
import matplotlib
matplotlib.use('agg')

import probscale
status = probscale.test(*sys.argv[1:])
sys.exit(status)
