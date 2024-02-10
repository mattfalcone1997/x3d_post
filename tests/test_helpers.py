import os
from shutil import rmtree
BASIC_ROOT = 'x3d_test_channel' 
BLAYER_ROOT = 'x3d_test_blayer' 

def _get_test_dirname(test_fn):
    dn = os.path.splitext(os.path.basename(test_fn))[0]
    return '_'.join([dn,'dir'])


def create_test_dir(test_fn):
    dn = _get_test_dirname(test_fn)
    if not os.path.isdir(dn):
        os.mkdir(dn)
    return dn

def destroy_test_dir(test_fn):
    dn = _get_test_dirname(test_fn)
    if os.path.isdir(dn):
        rmtree(dn)


