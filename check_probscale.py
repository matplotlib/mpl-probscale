import sys
import matplotlib
matplotlib.use('agg')

from matplotlib.pyplot import style
style.use('classic')

import probscale
status = probscale.test(*sys.argv[1:])
sys.exit(status)
