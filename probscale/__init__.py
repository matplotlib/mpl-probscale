from matplotlib import scale

from .viz import *
from .probscale import ProbScale
from .tests import test

scale.register_scale(ProbScale)
