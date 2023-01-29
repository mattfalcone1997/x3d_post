import flowpy as fp

from ._common import (CommonData,
                      CommonTemporalData)

from ._data_handlers import (stathandler_base,
                             stat_z_handler,
                             stat_xz_handler,
                             stat_xzt_handler)

from ._meta import meta_x3d
from ._average import (x3d_avg_xz,
                       x3d_avg_z,
                       x3d_avg_xzt)

from ..utils import get_iterations
from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_without_squeeze)
from itertools import product
import numpy as np
from ..style import get_symbol

_meta_class = meta_x3d
_avg_z_class = x3d_avg_z
_avg_xz_class = x3d_avg_xz
_avg_xzt_class = x3d_avg_xzt

class _quadrant_base(CommonData,stathandler_base):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if not from_hdf:
            self._quad_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)
    
    def _get_nstat(self,it):
        return (it - self.metaDF['initstat2']) // self.metaDF['istatcalc']
    
    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)
    
    @property
    def shape(self):
        return self._avg_data.shape

    @property
    def _coorddata(self):
        return self._avg_data._coorddata
        
    @property
    def Domain(self):
        return self._avg_data.Domain
            
    def save_hdf(self,fn,mode,key=None):
        key = self._get_hdf_key(key)
        
        hdf_obj = fp.hdfHandler(fn,mode,key=key)
        hdf_obj.set_type_id(self.__class__)
        
        self._meta_data.save_hdf(fn,'a',key+'/meta_data')
        self._avg_data.save_hdf(fn,'a',key+'/avg_data')
        
        self.quad_data.to_hdf(fn,'a',key=key+'/quad_data')
        
        return hdf_obj
    
    def _comp_calc(self,h,quadrant):
        return "h=%g Q%d"%(h,quadrant)
    
    def _check_quadrant(self,Quadrants):
        if hasattr(self,'Quadrants'):
            _quad_avail  = self.metaDF['h_quads']
        else:
            _quad_avail = [1,2,3,4]
            
        if Quadrants is None:
            Quadrants = _quad_avail

        else:
            Quadrants = fp.check_list_vals(Quadrants)
            if not all(quad in _quad_avail for quad in Quadrants):
                msg = f"The quadrants provided must be in {_quad_avail}"
                raise ValueError(msg)
        return Quadrants
        
    def _extract_uvq(self,path,it,it0):
        h_quads = self.metaDF['h_quads']
        
        l = self._get_data(path,'uv_quadrant_mean',None,it,len(h_quads)*4)

        if it0 is not None:
            l0 = self._get_data(path,'uv_quadrant_mean',None,it0,len(h_quads)*4)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)
            
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = [self._comp_calc(x,y) for x, y in product(h_quads,range(1,5))]
        index = self._get_index(it,comps)
        
        uvq = self._flowstruct_class(coorddata,
                                     l,
                                     index=index)
        return uvq

class x3d_quadrant_z(_quadrant_base,stat_z_handler):
    def _quad_extract(self,it,path,it0=None):
        self._meta_data = self._module._meta_class(path)
        
        self._avg_data = self._module._avg_z_class(it,path,it0)
        self.quad_data = self._extract_uvq(path,it,it0)
        
    def _hdf_extract(self,fn,key):
        key = self._get_hdf_key(key)
        
        hdf_obj = fp.hdfHandler(fn,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')
        self._avg_data = self._module._avg_z_class.from_hdf(fn,key=key+'/avg_data')
        
        self.quad_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/quad_data')
        
    def plot_line(self,h_list,coord, prop_dir,x_val=0,y_mode_wall=True,Quadrants=None,norm=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        if prop_dir == 'x':
            if y_mode_wall:
                CoordDF = self._avg_data.Wall_Coords(x_val)
                y = CoordDF.index_calc('y',coord)
            else:
                y = self.CoordDF.index_calc('y',coord)
                
            val = self.CoordDF['y'][y]
        else:
            val = self.CoordDF['x'][self.CoordDF.index_calc('x',coord)]
        
        if norm:
            time = max(self._avg_data.times)
            norm_Quadrant = self.quad_data/self._avg_data.uu_data[time,'uv']
        else:
            norm_Quadrant = self.quad_data
                    
        Quadrants = self._check_quadrant(Quadrants)
        quad_num = len(Quadrants)                    

        kwargs = update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)
        
        symbol = 'x' if prop_dir == 'y' else 'wall' if y_mode_wall else 'half-channel'
        unit = get_symbol(symbol)
        
        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = norm_Quadrant.plot_line(comp, prop_dir, val,
                                                    time = None,
                                                    labels=[f'$h = {h}$'],
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)

            ax[i].set_ylabel(r"$Q%d$"%quad)
        ax[0].set_title(r"$%s=%.5g$"%(unit,coord),loc='left')
        x_label = self.Domain.create_label(f"${prop_dir}$")
        ax[-1].set_xlabel(x_label)
        
        return fig, ax
    
class x3d_quadrant_xz(_quadrant_base,stat_xz_handler):
    def _quad_extract(self,it,path,it0=None):
        self._meta_data = self._module._meta_class(path)
        
        self._avg_data = self._module._avg_xz_class(it,path,it0)
        self.quad_data = self._extract_uvq(path,it,it0)

    def _hdf_extract(self,fn,key):
        key = self._get_hdf_key(key)
        
        hdf_obj = fp.hdfHandler(fn,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')
        self._avg_data = self._module._avg_xz_class.from_hdf(fn,key=key+'/avg_data')
        
        self.quad_data = fp.FlowStruct1D.from_hdf(fn,key=key+'/quad_data')

    def plot_line(self,h_list,Quadrants=None,norm=False,fig=None,ax=None,line_kw=None,**kwargs):
        if norm:
            time = max(self._avg_data.times)
            norm_Quadrant = self.quad_data/self._avg_data.uu_data[time,'uv']
        else:
            norm_Quadrant = self.quad_data
            
        Quadrants = self._check_quadrant(Quadrants)
        quad_num = len(Quadrants)

        kwargs = update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)
            
        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = norm_Quadrant.plot_line(comp,
                                                    time = None,
                                                    label=f'$h = {h}$',
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)
            ax[i].set_ylabel(r"$Q%d$"%quad)

        x_label = self.Domain.create_label(f"$y$")
        ax[-1].set_xlabel(x_label)

        return fig, ax
            
class x3d_quadrant_xzt(x3d_quadrant_xz,CommonTemporalData,stat_xzt_handler):
    def _quad_extract(self,path,its=None):
        if its is None:
            its = get_iterations(path,statistics=True)
            
        self._meta_data = self._module._meta_class(path)
        
        self._avg_data = self._module._avg_xzt_class(path,its=its)
        self.quad_data = self._extract_uvq(path,its,None)
    
    @property
    def times(self):
        return self.quad_data.times
    
    def _hdf_extract(self,fn,key):
        key = self._get_hdf_key(key)
        
        hdf_obj = fp.hdfHandler(fn,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')
        self._avg_data = self._module._avg_xzt_class.from_hdf(fn,key=key+'/avg_data')
        
        self.quad_data = fp.FlowStruct1D_time.from_hdf(fn,key=key+'/quad_data')
        
    def plot_line(self, h_list,prop_dir ,coord, y_mode_wall=True, Quadrants=None, norm=False, fig=None, ax=None, line_kw=None, **kwargs):
        Quadrants = self._check_quadrant(Quadrants)

        if norm:
            comps = [self._comp_calc(h,quad) for h, quad in product(h_list,Quadrants)]
            times = self.times if prop_dir == 't' else [coord]
            norm_Quadrant = self.quad_data[times,comps].copy()
            for time in times:
                for comp in comps:
                    norm_Quadrant[time,comp] /= self._avg_data.uu_data[time,'uv']
        else:
            norm_Quadrant = self.quad_data
            
        quad_num = len(Quadrants)

        kwargs = update_subplots_kw(kwargs,figsize=[12,5])
        fig, ax, _ = create_fig_ax_without_squeeze(quad_num,1,fig=fig,ax=ax,**kwargs)
        ax = ax.reshape(quad_num)
        if prop_dir == 't':
            symbol = 'x' if prop_dir == 'y' else 'wall' if y_mode_wall else 'half-channel'
            unit = get_symbol(symbol)
            if y_mode_wall:
                CoordDF = self._avg_data.Wall_Coords(coord)
                y = CoordDF.index_calc('y',coord)
            else:
                y = self.CoordDF.index_calc('y',coord)
                
            val = self.CoordDF['y'][y]
            
            for i, quad  in enumerate(Quadrants):
                for h in h_list:
                    comp = self._comp_calc(h,quad)
                    fig, ax[i] = norm_Quadrant.plot_line_time(comp, val,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i],
                                                            line_kw=line_kw)
                ax[i].set_ylabel(r"$Q%d$"%quad)
            ax[0].set_title(r"$%s=%.3g$"%(unit,coord),loc='left')
            x_label = self.Domain.create_label(f"$t$")
            ax[-1].set_xlabel(x_label)
            
        else:
            for i, quad  in enumerate(Quadrants):
                for h in h_list:
                    comp = self._comp_calc(h,quad)
                    fig, ax[i] = norm_Quadrant.plot_line(comp,
                                                        time = coord,
                                                        label=f'$h = {h}$',
                                                        channel_half = True,
                                                        fig= fig, ax= ax[i],
                                                        line_kw=line_kw)
                    
            x_label = self.Domain.create_label(f"$y$")
            ax[-1].set_xlabel(x_label)
            title = self.Domain.create_label(r"$t=%.3g$"%coord)
            ax[0].set_title(title,loc='left')
                
            ax[i].set_ylabel(r"$Q%d$"%quad)
                
        return fig, ax
    