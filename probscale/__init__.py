from matplotlib import scale

from .viz import *
from .probscale import ProbScale

scale.register_scale(ProbScale)

from numpy.testing import Tester
test = Tester().test