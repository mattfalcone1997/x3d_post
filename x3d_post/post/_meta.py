from builtins import Exception, object
from ._data_handlers import read_parameters
from ._common import Common
import numpy as np
import flowpy as fp
import json
from os.path import join, isfile
import pandas as pd

class meta_x3d(Common):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._meta_extract(*args,**kwargs)

    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)

    @property
    def coorddata(self):
        return fp.AxisData(fp.GeomHandler(self.metaDF['itype']),
                            self.CoordDF,
                            coord_nd=None)

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
                           itype=itype,
                           istatcalc=params.get('istatcalc'),
                           initstat=params.get('initstat'),
                           dt=params.get('dt'))

        if 'uv_quadrant' in params:
            h_quads = params['uv_quadrant']['h_quads']
            self.metaDF['h_quads'] = h_quads
        
        if 'autocorrelation' in params:
            shape = params['autocorrelation']['shape']
            x_locs = params['autocorrelation']['x_locs']
            
            self.metaDF['autocorr_shape'] = shape
            self.metaDF['autocorr_x_locs'] = x_locs
            
        self.Domain = fp.GeomHandler(itype)

        self._extract_run_params(path)
        
        self._meta_hook(params) 
               
    def _extract_run_params(self,path):
        fn = join(path,'run_log.json')
        if not isfile(fn): 
            return
        
        params = pd.read_json(fn,orient='index')
        its = params['t0']
    
        while not all(np.diff(its[::-1])<0):
            ind = ~(np.diff(its[::-1])<0)
            index = params.index[[*ind[::-1],False]]
            params.drop(index=index,inplace=True)
            its = params['t0']
        
        
        self._run_params = params.pivot_table(index='itime0')
    
    def get_it(self,time):
        if not hasattr(self,'_run_params'):
            return time/self.metaDF['dt']
        
        ts = self._run_params['t0']
        it0 = self._run_params.index[ts<time][-1]
        t0 = self._run_params['t0'][it0]
        dt = self._run_params['dt'][it0]
        
        return it0 + (time - t0)/dt
    
    def get_time(self, it):
        if not hasattr(self,'_run_params'):
            return it*self.metaDF['dt']
        
        index = self._run_params.index
        it0 = self._run_params.index[it>index][-1]
        t0 = self._run_params['t0'][it0]
        dt = self._run_params['dt'][it0]
        return t0 + (it - it0 + 1)*dt
    
    def get_dt(self,*,it=None,time=None):
        
        if it is None and time is None:
            raise ValueError("it and time cannot both be None")
        
        if it is None:
            it = self.get_it(time)
        
        index = self._run_params.index
        it0 = self._run_params.index[it>index][-1]
        return self._run_params['dt'][it0]
        
    def _meta_hook(self,params):
        pass
    
    def _hdf_extract(self,fn, key=None):
        key = self._get_hdf_key(key)

        self.CoordDF = fp.coordstruct.from_hdf(fn,key=key+'/CoordDF')

        h5_obj = fp.hdfHandler(fn,'r',key=key)
        self.NCL = h5_obj['NCL'][:]
        self.metaDF = dict(**h5_obj['metaDF'].attrs)
        self.Domain = fp.GeomHandler(self.metaDF['itype'])
        if 'run_params' in h5_obj.keys():
            self._run_params = pd.read_hdf(fn,key=key+'/run_params')
        return h5_obj

    def save_hdf(self,fn,mode,key=None):
        key = self._get_hdf_key(key)

        h5_obj = fp.hdfHandler(fn,mode,key=key)

        h5_obj.create_dataset('NCL',data=self.NCL)
        self.CoordDF.to_hdf(fn,'a',key=key+'/CoordDF')

        metadf = h5_obj.create_group('metaDF')
        for k, v in self.metaDF.items():
            if v is not None:
                metadf.attrs[k] = v
        
        if hasattr(self,'_run_params'):
            self._run_params.to_hdf(fn,key=key+'run_params')         

        return h5_obj

