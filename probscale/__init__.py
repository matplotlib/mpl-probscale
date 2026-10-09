from matplotlib import scale

from .probscale import ProbScale
from .viz import *

scale.register_scale(ProbScale)


__version__ = "0.2.6dev"
__author__ = "Paul Hobson (Herrera Environmental Consultants)"
__author_email__ = "phobson@herrerainc.com"
