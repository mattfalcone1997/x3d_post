import x3d_post.post as xp
import os
import json
import numpy as np
import pytest
from test_helpers import (BASIC_ROOT,
                          create_test_dir,
                          destroy_test_dir)

dn = create_test_dir(__file__)
path = BASIC_ROOT

meta_data = xp.meta_x3d(path)

h5_fn = os.path.join(dn,'meta.h5')

# compare against json
with open(os.path.join(path,'parameters.json')) as f:
    params = json.loads(f.read())

ncl = meta_data.NCL

assert ncl[0] == params['mesh']['sizes'][0]
assert ncl[1] == params['mesh']['sizes'][1]
assert ncl[2] == params['mesh']['sizes'][2]

x = params['mesh']['xcoords']
y = params['mesh']['ycoords']
z = params['mesh']['zcoords']

assert all(x == meta_data.CoordDF['x'])
assert all(y == meta_data.CoordDF['y'])
assert all(z == meta_data.CoordDF['z'])

# test saving to hdf5 file format

meta_data.save_hdf(h5_fn,'w')

meta_data1 = xp.meta_x3d.from_hdf(h5_fn)


for k in meta_data.__dict__:
    test = meta_data.__dict__[k] != meta_data1.__dict__[k]
    if hasattr(test,'__iter__'):
        test = all(test)

    if test:
        pytest.fail(f'Atrributes of saved meta_x3d object do not match: {k}')
        
avg_data = xp.x3d_avg_z(25,path)

avg_data.save_hdf(os.path.join(dn,'avg_data.h5'),'w')

avg_data1 = xp.x3d_avg_z.from_hdf(os.path.join(dn,'avg_data.h5'))

for k in avg_data.__dict__:
    test = avg_data.__dict__[k] != avg_data1.__dict__[k]
    if hasattr(test,'__iter__'):
        test = all(test)

    if test:
        print(f'Atrributes of saved x3d_avg_z object do not match: {k}')


budget = xp.x3d_budget_z('uu',25,path)
budget.save_hdf(os.path.join(dn,'budget.h5'),'w')
budget1 = xp.x3d_budget_z.from_hdf(os.path.join(dn,'budget.h5'))

for k in budget.__dict__:
    test = budget.__dict__[k] != budget1.__dict__[k]
    if hasattr(test,'__iter__'):
        test = all(test)

    if test:
        print(f'Atrributes of saved x3d_budget_z object do not match: {k}')


destroy_test_dir(__file__)