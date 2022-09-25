from ._data_handlers import read_parameters
from ._common import Common
import numpy as np
import flowpy as fp

class meta_x3d(Common):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._meta_extract(*args,**kwargs)

    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)

    def _meta_extract(self,path):
        params = read_parameters(path)
        coords = {'x' : np.array(params['mesh']['xcoords']),
                  'y' : np.array(params['mesh']['ycoords']),
                  'z' : np.array(params['mesh']['zcoords'])}

        self.CoordDF = fp.coordstruct(coords)
        self.NCL = params['mesh']['sizes']
        if params['itype'] == 3:
            itype = fp.CHANNEL
        elif params['itype'] == 13:
            itype = fp.BLAYER
        else:
            raise RuntimeError("Other flow types not yet checked")

        self.metaDF = dict(re=params['re'],
                           dt = params['dt'],
                           itype=itype,
                           istatcalc=params.get('istatcalc'),
                           initstat=params.get('initstat'))
        
    def _hdf_extract(self,fn, key=None):
        key = self._get_hdf_key(key)

        self.CoordDF = fp.coordstruct.from_hdf(fn,key=key+'/CoordDF')

        h5_obj = fp.hdfHandler(fn,'r',key=key)
        self.NCL = h5_obj['NCL'][:]
        self.metaDF = dict(**h5_obj['metaDF'].attrs)
        return h5_obj

    def save_hdf(self,fn,mode,key=None):
        key = self._get_hdf_key(key)

        h5_obj = fp.hdfHandler(fn,mode,key=key)

        h5_obj.create_dataset('NCL',data=self.NCL)
        metadf = h5_obj.create_group('metaDF')

        self.CoordDF.to_hdf(fn,'a',key=key+'/CoordDF')

        for k, v in self.metaDF.items():
            if v is not None:
                metadf.attrs[k] = v

        return h5_obj

