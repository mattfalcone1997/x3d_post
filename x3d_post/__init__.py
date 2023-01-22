from flowpy import rcParams

from .style import *
from matplotlib.rcsetup import validate_bool

rcParams.validate.update({'new_data' : validate_bool,
                          'correct_gradients' : validate_bool})
dict.update(rcParams,{'new_data':True,
                      'correct_gradients' : True})