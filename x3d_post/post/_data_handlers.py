import flowpy as fp
import numpy as np
from abc import ABC, abstractmethod
from os.path import join
import json
import xml.etree.ElementTree as ET
import os
from ..utils import check_path
import time

from numbers import Number
def read_parameters(path):
    fn = join(path,'parameters.json')
    with open(fn,'r') as f:
        params = json.load(f)

    return params

def read_stat_z_file(file_path,shape,dtype='f8',mean_x=False):
    data = np.fromfile(file_path,dtype=dtype).reshape(shape)
    if mean_x:
        data = data.mean(axis=-1)
    return data

class stathandler_base(ABC):
    _flowstruct_class = None
    
    @staticmethod
    def _get_stat_file_z(path,name,it):
        check_path(path,statistics=True)

        stat_path = os.path.join(path,'statistics')

        fn = name + '.dat'+ str(it).zfill(7)

        return os.path.join(stat_path,fn)
    
    def _fix_dumb_error(self,comps: list[str],axis,data,check,factor):
        slicer = [slice(None)]*(data.ndim-1)
        if axis is not None:
            slicer[axis] = slice(1,-1)
        for i, comp in enumerate(comps):
            n = comp.count(check)
            if n > 0:
                multiplier = factor**n
                data[i][tuple(slicer)] *= multiplier
        return data
    
    def _apply_symmetry(self,comp: str,array: np.ndarray,axis: int,noflip: list[str]= None,exclude: list[str]= None):

        if exclude is not None:
            if comp in exclude:
                return array
              
        factor = (-1)**(comp.count('v')+comp.count('y'))
        if noflip is not None:
            if comp in noflip:
                factor = 1.0

        slicer = [slice(None)]*array.ndim
        slicer[axis] = slice(None,None,-1)
    
        return 0.5*(array + factor*array[tuple(slicer)])
        
        
    def _check_attr(self,attr):
        if not hasattr(self,attr):
            raise AttributeError(f"Attribute {attr} must be "
                            "created to call this function")

    @classmethod
    def avg_avail(cls,it,path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory {path} not found")
        
        stat_path = os.path.join(path,'statistics')
        fn = os.path.join(stat_path,'umean.dat'+ str(it).zfill(7))

        return os.path.isfile(fn)
    
    def _get_index(self,it,comps):
        
        if isinstance(it,Number):
            it = [it]
            
        times = np.array([self._meta_data.get_time(i) for i in it])
        times = (times[:,None]*np.ones(len(comps))).flatten()
        comps = list(comps)*len(it)
        return [times,comps]
    
    @abstractmethod
    def _get_old_data(self,path,names,it):
        pass
    
    @abstractmethod
    def _get_new_data(self,path,name,it,size):
        pass
    
    def _correct_mean_gradients(self,fpath,data,comps):
        
        REF_DATE = [2023,1,13,15,0,0]

        mod_date = time.localtime(os.stat(fpath).st_mtime)[:6]
        try:
            check1 = [a>b for a, b in zip(REF_DATE,mod_date)].index(True)
        except ValueError:
            check1 = 7
        try:
            check2 = [a<b for a, b in zip(REF_DATE,mod_date)].index(True)
        except ValueError:
            check2 = 7
        
        
        make_correct =  check2 > check1
                
        if fp.rcParams['correct_gradients'] and make_correct:
            print("x3d_post is correcting mean gradient calculations")
            data = self._fix_dumb_error(comps,1,data,'dx',0.5)
            factor = 0.5*(self.NCL[0]/(self.NCL[0]-2.))
            data = self._fix_dumb_error(comps,None,data,'dz',factor)
            
        return data

    def _get_data(self,path,name,old_names,it,size,comps=None):
        if fp.rcParams['new_data']:
            data = self._get_new_data(path,name,it,size)
        else:
            data = self._get_old_data(path,old_names,it)
        
        fpath = self._get_stat_file_z(path,name,it)

                
        if comps is not None:
            data = self._correct_mean_gradients(fpath,data,comps)
            
        return data
    def _get_nstat(self,it):
        return (it - self.metaDF['initstat']) // self.metaDF['istatcalc']


class stat_z_handler(stathandler_base,ABC):
    _flowstruct_class = fp.FlowStruct2D
    def _get_old_data(self,path,names,it):
        shape = (len(names), self.NCL[1], self.NCL[0])

        l = np.zeros(shape)
        for i, name in enumerate(names):
            fn = self._get_stat_file_z(path,name,it)
            l[i] = read_stat_z_file(fn,shape[1:])
        return l
    
    def _get_new_data(self,path,name,it,size):
        shape = (size, self.NCL[1], self.NCL[0])
        fn = self._get_stat_file_z(path,name,it)
        return read_stat_z_file(fn,shape)
    
class stat_xz_handler(stat_z_handler,ABC):
    _flowstruct_class = fp.FlowStruct1D
    def _get_data(self,*args,**kwargs):
        l = super()._get_data(*args,**kwargs)
        return l.mean(axis=-1)        

class stat_xzt_handler(stat_xz_handler,ABC):
    _flowstruct_class = fp.FlowStruct1D_time
    def _get_old_data(self,path,names,its):
           
        shape = (len(names)*len(its), self.NCL[1],self.NCL[0])

        l = np.zeros(shape)
        i = 0
        for it in its:
            for name in names:
                fn = self._get_stat_file_z(path,name,it)
                l[i] = read_stat_z_file(fn,shape[1:])
                i += 1
        return l
    
    def _get_new_data(self,path,name,its,size):
        shape = (size*len(its), self.NCL[1],self.NCL[0])
        l = np.zeros(shape)
        for i, it in enumerate(its):
            l[i*size:(i+1)*size] = super()._get_new_data(path,name,it,size)
            
        return l

class inst_reader(ABC):
    _reader_comps = {'u':'ux',
                     'v':'uy',
                     'w':'uz',
                     'p':'pp',}
    _default_comps = ['u','v','w','p']
    def _check_comps(self,comps):
        if comps is None:
            return self._default_comps

    def _extract_xml(self,fn):
        root =ET.parse(fn).getroot()

        shape_str = root.find('Domain/Topology').get('Dimensions')
        shape = tuple(int(s) for s in shape_str.split())

        geom = root.find('Domain/Geometry').findall('DataItem')

        geom_data = [None]*3

        data = geom[2].text.replace('\n','').split()
        geom_data[0] = np.array([np.float64(x) for x in data])

        data = geom[1].text.replace('\n','').split()
        geom_data[1] = np.array([np.float64(x) for x in data])

        data = geom[0].text.replace('\n','').split()
        geom_data[2] = np.array([np.float64(x) for x in data])

        return shape, geom_data

    def _extract_inst_xdmf(self,it,path,comps=None):
        data_folder = join(path,'data')

        self._check_comps(comps)

        xml_fn = join(data_folder,'snapshot-%s.xdmf'%str(it).zfill(7))
        shape, geom_data = self._extract_xml(xml_fn)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coords = fp.coordstruct({'x':geom_data[2],
                                 'y':geom_data[1],
                                 'z':geom_data[0]})

        coorddata = fp.AxisData(geom, coords, coord_nd=None)

        l =[]
        for comp in comps:
            c = self._reader_comps[comp]
            fn = join(data_folder,'%s-%s.bin'%(c,str(it).zfill(7)))

            data = np.fromfile(fn,dtype='f8').reshape(shape)
            l.append(data)
        time = self._meta_data.get_time(it)
        index = [[time]*len(comps),comps]
        u_data = fp.FlowStruct3D(coorddata,
                                  np.array(l),
                                  index=index)

        return u_data




            