from abc import ABC, abstractproperty
import warnings
import sys
from ..utils import check_path
import os


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
    def metaDF(self):
        return self._meta_data.metaDF

    @property
    def CoordDF(self):
        return self._meta_data.CoordDF

    @property
    def NCL(self):
        return self._meta_data.NCL

    @abstractproperty
    def Domain(self):
        pass

    @abstractproperty
    def _coorddata(self):
        pass

    def _get_stat_file_z(self,path,name,it):
        check_path(path,statistics=True)

        stat_path = os.path.join(path,'statistics')

        fn = name + '.dat'+ str(it).zfill(7)

        return os.path.join(stat_path,fn)

