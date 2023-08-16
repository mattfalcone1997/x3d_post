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
            if os.path.isfile(fn):
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
    
    def _extract_probe_file(self, probe_path: str, fn:str, n:int, its_req: list)->dict[int,fp.FlowStructND_time]:
        if len(its_req) == 0:
            return None
        
        with open(join(probe_path,fn),'r') as f:
                probe_info = json.load(f)
            
        its = np.loadtxt(join(probe_path,f'probe_times{n}.csv'),
                            delimiter=',',skiprows=1,dtype='i4',usecols=0)
        times = np.loadtxt(join(probe_path,f'probe_times{n}.csv'),
                            delimiter=',',skiprows=1,dtype='f8',usecols=1)

        ind = [it in its_req for it in its]
        if not any(ind):
            raise RuntimeError("Their is an internal issue here")

        nprobes = probe_info['nlineprobes']
        fn_list = ['lineprobe%.4d-run%d'%(i+1,n) for i in range(nprobes)]
        
        probe_data = {}
        shape = (len(times),3,self.meta_data.NCL[2])
        index = list(product(times[ind],['u','v','w']))
        shape1 = (len(times[ind])*3,self.meta_data.NCL[2])
        
        coorddata = fp.AxisData(self.meta_data.Domain,
                                    self.meta_data.CoordDF[['z']],
                                    None)
        
        for i, fn in enumerate(fn_list):
            data = np.fromfile(join(probe_path,fn)).reshape(shape)
            probe_data[i+1] = fp.FlowStructND_time(coorddata,data[ind,:,:].reshape((shape1)),index=index,data_layout=('z'))
            
        return probe_data

    def _extract_probes(self,path):
        
        probe_path = join(path,'probes')
        params_files = sorted([fn for fn in os.listdir(probe_path)\
                        if 'probe_info' in fn])
        
        self.meta_data = self._module._meta_class(path)
        probe_params = [json.loads(open(os.path.join(probe_path,f),'r',encoding='ascii').read())\
                         for f in params_files]
        self._check_probes(probe_params)

        if len(params_files) > 1:
            
            its_list = self._get_probe_times(probe_path)
            
            probe_datas = [self._extract_probe_file(probe_path,fn,i+1,its_list[i])\
                          for i, fn in enumerate(params_files) ]
            while None in probe_datas:
                probe_datas.remove(None)

            probe_data = probe_datas[0]
            for probe in probe_datas[1:]:
                for i in range(len(probe_data)):
                    probe_data[i+1].concat(probe[i+1])

            self.probe_data = probe_data
        else:
            self.probe_data = self._extract_probe_file(probe_path,params_files[0],1)
    
    def _get_probe_times(self,probe_path):
        times_files = [os.path.join(probe_path,fn)\
                        for fn in os.listdir(probe_path)\
                              if 'probe_times' in fn]
        
        its_list = [np.loadtxt(fn,skiprows=1,delimiter=',',usecols=0,dtype='i4') for fn in sorted(times_files)]

        registered_its = list(its_list[-1])
        its_required =[its_list[-1]]
        for its in its_list[:-1][::-1]:
            added_its = [it for it in its if it not in registered_its]
            its_required.append(added_its)
            registered_its.extend(added_its)

        return its_required[::-1]
            
    def _check_probes(self,probe_params: list[dict]):
        probes_run = []
        for run in probe_params:
            probes_run.append([v for k, v in run.items() if 'probe ' in k ])
        
        if len(probes_run) > 1:
            for probe in probes_run[1:]:
                if not np.allclose(probes_run[0],probe):
                    raise RuntimeError("Probe locations are not close")
           
        self.probe_locations = {}
        for i in range(len(probes_run[0])):
            self.probe_locations[i+1] = probe_params[0][f'probe {i+1}']

    def save_hdf(self,fn,mode,key=None):
        if key is None:
            key = self._get_hdf_key(key)

        hdf_obj = fp.hdfHandler(fn,mode,key=key)
        self.meta_data.save_hdf(fn,'a',key=key+'/meta_data')
        
        for k,v in self.probe_data.items():
            v.to_hdf(fn,'a',key=key+'/probe %d'%k)        
            hdf_obj['probe %d'%k].attrs['location'] =self.probe_locations[k]
    
    def _hdf_extract(self,fn,key=None):
        if key is None:
            key = self._get_hdf_key(key)
            
        self.meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')
        
        h5_obj = fp.hdfHandler(fn,'r',key=key)
        keys = sorted([int(k.split()[-1]) for k in h5_obj.keys() if 'probe' in k])

        self.probe_data = {}
        self.probe_locations = {}
        for k in keys:
            self.probe_data[k] = fp.FlowStructND_time.from_hdf(fn,key=key+'/probe %d'%k)
            self.probe_locations[k] = h5_obj['probe %d'%k].attrs['location']
