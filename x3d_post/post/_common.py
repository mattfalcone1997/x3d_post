from abc import ABC, abstractproperty
import warnings
import sys
from ..utils import check_path, get_iterations
import os
import numpy as np
import flowpy as fp
from inspect import getmembers
from functools import wraps
class classproperty():
    def __init__(self,func):
        self.f = func
    def __get__(self,obj,cls):
        return self.f(cls)

class _classfinder:
    def __init__(self,cls):
        self._cls = cls
    
    def __getattr__(self,attr):
        mro = self._cls.mro()
        
        for c in mro:
            module = sys.modules[c.__module__]
            if hasattr(c,attr):
                return getattr(self._cls,attr)
            elif hasattr(module, attr) and hasattr(c,'_module'):
                if c in mro[1:]:
                    warn_msg = (f"Attribute {attr} being inherited "
                                f"from parent class ({c.__module__}.{c.__name__})"
                                 ". This behavior may be undesired")
                    warnings.warn(warn_msg)
                    
                return getattr(module,attr)
        msg = f"Attribute {attr} was not found for class {mro[0].__name__} in module {mro[0].__module__}"
        raise ModuleNotFoundError(msg) 

class Common(ABC):

    @classproperty
    def _module(cls):
        return _classfinder(cls)

    def _get_hdf_key(self,key):
        if key is None:
            return self.__class__.__name__
        else:
            return key

class CommonData(Common):
    @property
    def metaDF(self) -> dict:
        return self._meta_data.metaDF

    @property
    def CoordDF(self) -> fp.coordstruct:
        return self._meta_data.CoordDF

    @property
    def NCL(self) -> np.ndarray:
        return self._meta_data.NCL

    @abstractproperty
    def Domain(self) -> fp.GeomHandler:
        pass

    @abstractproperty
    def _coorddata(self) -> fp.AxisData:
        pass

    @staticmethod
    def _get_stat_file_z(path,name,it):
        check_path(path,statistics=True)

        stat_path = os.path.join(path,'statistics')

        fn = name + '.dat'+ str(it).zfill(7)

        return os.path.join(stat_path,fn)

    @property
    def _flowstructs(self) -> dict[fp.FlowStruct1D]:
        members = getmembers(self,lambda x: isinstance(x,fp.FlowStructND))
        return dict(members)

def require_override(func):
    
    @wraps(func)
    def _return_func(*args,**kwargs):
       msg = (f"Method {__name__} must be overriden to be used. There is no "
              "need to override it if this function is not called")
       raise NotImplementedError(msg)
   
    return _return_func

class CommonTemporalData(CommonData):
    @classmethod
    def phase_average(cls,*objects,items=None):
        if not all(type(x)==cls for x in objects):
            msg = (f"All objects to be averaged must be of type {cls.__name.__}"
                    f" not {[type(x).__name__ for x in objects]}")
            raise TypeError(msg)
        
        if items is not None:
            if len(items) != len(objects):
                msg = ("If items is present, it must be the same"
                       " length as the inputs to be phased averaged")
                raise ValueError(msg)
            items = np.array(items)
        else:
            items = np.ones(len(objects))

        base_object = objects[0].copy()
        object_attrs = dir(base_object)
        
        for attr in object_attrs:
            vals = [getattr(ob,attr) for ob in objects]
            val_type = type(vals[0])
            
            if not all(type(val) == val_type for val in vals):
                msg = ("Not all attributes of object "
                       "to be phased averaged are of the same type")
                raise TypeError(msg)
            
            if issubclass(val_type,CommonTemporalData):
                setattr(base_object,
                        attr,
                        val_type.phase_average(*vals,
                                               items=items))
            elif issubclass(val_type,fp.FlowStructND):
                    
                time_shifts = [x._time_shift for x in objects]
                vals = [val.copy().shift_times(shift) \
                            for val,shift in zip(vals,time_shifts)]
                
                times_list = [val.times for val in vals]

                vals = base_object._handle_time_remove(vals,times_list)
                coeffs = items/np.sum(items)
                phase_val = sum(coeffs*vals)
                
                setattr(base_object,attr,phase_val)
                
            else:
                cls._type_hook(base_object,attr,vals)
    
    def _type_hook(cls,base_object,attr,vals):
        pass
    
    @abstractproperty
    def times(self):
        pass
    
    def set_times(self,value):
        for  v in self._flowstructs.values():
            if not len(value) == len(v.times):
                raise ValueError("The length of the new times"
                                " must be the same as the existing one")
            v.times = value
            
    def _del_times(self,times):
        for  v in self._flowstructs.values():
            for time in times:
                v.remove_time(time)
                
    def shift_times(self,time):
        for v in self._flowstructs.values():
            v.shift_times(time)

    def _handle_time_remove(self,fstructs: list[fp.FlowStructND],
                                times_list: list):
        
        for i, fstruct in enumerate(fstructs):
            intersect_times = self._get_intersect(times_list)

            for time in fstruct.times:
                if time not in intersect_times:
                    fstruct.remove_time(time)
                
        return fstructs
    
    @classmethod
    def _get_its_phase(cls,paths,times=None) -> list[int]:
        it_shifts = [cls._get_its_shift(path) for path in paths]

        if times is None:
            
            times_list = [ np.array(get_iterations(path,statistics=True)) + shift\
                        for shift, path in zip(it_shifts,paths)]
            
            times_shifted = cls._get_intersect(times_list)
        else:
            times_shifted = times    
                
        return [times_shifted - shift for shift in it_shifts]

    @classmethod
    def _get_intersect(cls,its_list):
        base_set = set(its_list[0])
        for its in its_list[1:]:
            base_set.intersection_update(its)
        
        return list(base_set)
            
    @require_override
    def _time_shift(self):
        pass
    
    @require_override
    def _get_its_shift(cls,path) -> int:
        pass
    
    def _test_times_shift(self,path):
        
        time_shift1 = self._get_its_shift(path)*self.metaDF['dt']
        time_shift2 = self._time_shift
        
        if time_shift1 != time_shift2:
            msg = ("methods _get_times_shift and"
                   " _time_shift must return the same value."
                   f" Current_values: {time_shift1} {time_shift2}")
            raise RuntimeError(msg)    