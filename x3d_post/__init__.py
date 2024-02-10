from flowpy import rcParams

from .style import *
from matplotlib.rcsetup import validate_bool, validate_int_or_None

rcParams.validate.update({'new_data' : validate_bool,
                          'correct_gradients' : validate_bool,
                          'file_workers':validate_int_or_None})
dict.update(rcParams,{'new_data':True,
                      'correct_gradients' : True,
                      'file_workers':None})