from ._data_handlers import read_parameters
from ._common import Common
import numpy as np
import flowpy as fp
import json
from os.path import join, isfile
import os
import pandas as pd
from itertools import product

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
        elif params['itype'] == 13 or params['itype']==14:
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
        
        if 'window' in params:
            fn = params['window']['file']
            data = np.loadtxt(os.path.join(path,fn)).squeeze()
            self.metaDF['outputs'] = data
            
        if 'autocorrelation' in params:
            shape = params['autocorrelation']['shape']
            x_locs = params['autocorrelation']['x_locs']
            
            self.metaDF['autocorr_shape'] = shape
            self.metaDF['autocorr_x_locs'] = x_locs
            
        if 'spectra_corr' in params:
            ylocs = params['spectra_corr']['y_locs']
            self.metaDF['spectra_ylocs'] = ylocs
            
        self.Domain = fp.GeomHandler(itype)

        self._extract_run_params(path)
        
        self._meta_hook(path,params) 
               
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
        
    def _meta_hook(self,path,params):
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

_meta_class = meta_x3d
class probes(Common):
    pass

class line_probes(Common):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_probes(*args,**kwargs)
            
    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)
    
    def _extract_probes(self,path):
        
        probe_path = join(path,'probes')
        params_files = [fn for fn in os.listdir(probe_path)\
                        if 'probe_info' in fn]
        self.meta_data = self._module._meta_class(path)
        if len(params_files) > 1:
            raise NotImplementedError("May be done if necessary")
            param_list = []
            for fn in params_files:
                param_list.append(json.loads(join(probe_path,fn)))

            if self._check_probes(param_list):
                raise ValueError('Run no must be provided if probes are not the same')
            probe_info = param_list[0]
            
            times_list = [join(probe_path,'probes_times%d.csv'%(i+1)) \
                            for i in range(len(param_list))]
        else:
            with open(join(probe_path,params_files[0]),'r') as f:
                probe_info = json.load(f)
            
            times = np.loadtxt(join(probe_path,'probe_times1.csv'),usecols=1,
                               delimiter=',',skiprows=1)
            nprobes = probe_info['nlineprobes']
            fn_list = ['lineprobe%.4d-run1'%(i+1) for i in range(nprobes)]
            
            probe_data = {}
            shape = (len(times)*3,self.meta_data.NCL[2])
            index = list(product(times,['u','v','w']))
            
            coorddata = fp.AxisData(self.meta_data.Domain,
                                       self.meta_data.CoordDF,
                                       None)
            
            for i, fn in enumerate(fn_list):
                data = np.fromfile(join(probe_path,fn)).reshape(shape)
                probe_data[i+1] = fp.FlowStructND_time(coorddata,data,index=index,data_layout=('z'))
                
            self.probe_data = probe_data
            
            
    def _check_probes(self,probe_params: list[dict]):
        probes_run = []
        for run in probe_params:
            probes_run.append({k:v for k,v in run.items() if 'probe' in k })
            
        for probe in probes_run[1:]:
           if not np.isclose(probes_run[0]['x'],probe['x']):
               return False
           if not np.isclose(probes_run[0]['y'],probe['y']):
               return False
           
        return True
            
    def save_hdf(self,fn,mode,key=None):
        if key is None:
            key = self._get_hdf_key(key)
            
        self.meta_data.save_hdf(fn,mode,key=key+'/meta_data')
        
        for k,v in self.probe_data.items():
            v.to_hdf(fn,'a',key=key+'/probe %d'%k)        
    
    def _hdf_extract(self,fn,key=None):
        if key is None:
            key = self._get_hdf_key(key)
            
        self.meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')
        
        h5_obj = fp.hdfHandler(fn,'r',key=key)
        keys = [k for k in h5_obj.keys() if 'probe' in k]
        self.probe_data = {}
        for i,k in enumerate(keys):
            self.probe_data[i+1] = fp.FlowStructND_time.from_hdf(fn,key=key+'/'+k)