
from abc import abstractmethod
import numpy as np
import flowpy as fp
from flowpy.utils import check_list_vals
from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_without_squeeze,
                             create_fig_ax_with_squeeze)
from ..style import get_symbol
import matplotlib as mpl
from ._meta import meta_x3d
_meta_class=meta_x3d

from ._average import x3d_avg_z
from ._common import CommonData

_avg_z_class = x3d_avg_z

from ._instant import x3d_inst_z
_inst_z_class = x3d_inst_z

class _fluct_base(CommonData):

    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._fluct_extract(*args,**kwargs)

    @abstractmethod
    def _hdf_extract(self,*args,**kwargs):
        pass

    @abstractmethod
    def _fluct_extract(self,*args,**kwargs):
        pass
    
    @property
    def _meta_data(self):
        return self.avg_data._meta_data
    
    @property
    def _coorddata(self):
        return self.fluct_data._coorddata

    @property
    def Domain(self):
        return self.fluct_data.Domain

    @classmethod
    def from_hdf(cls,file_name,key=None):
        return cls(file_name,from_hdf=True,key=key)

    def save_hdf(self,file_name,write_mode,key=None):
        key = self._get_hdf_key(key)

        hdf_obj = fp.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self.avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self.fluct_data.to_hdf(file_name,key=key+'/fluctdata',mode='a')


    @property
    def shape(self):
        return self.fluct_data.shape
    
    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = f"PhyTime provided ({PhyTime}) is not in the {self.__class__.__name__} datastruct, recovery impossible"
        
        err = ValueError(err_msg)
        warn = UserWarning(warn_msg)
        return self.fluct_data.check_times(PhyTime,err_msg,warn_msg)
    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,wall_units=True,fig=None,ax=None,contour_kw=None,**kwargs):
        
        axis_vals = check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.fluct_data.CoordDF.check_plane(plane)

        if coord == 'y' and wall_units:
            int_vals = self.avg_data.get_coords_wall_units(coord,axis_vals,0)           
            axis_vals = self.avg_data.Wall_Coords(0).get_true_coords('y',axis_vals)
            title_symbol = get_symbol('wall_initial')

        else:
            int_vals = axis_vals = self.CoordDF.get_true_coords(coord,axis_vals)
            title_symbol = self.Domain.create_label(coord)

        x_size, z_size = self.fluct_data.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)


        for i,val in enumerate(int_vals):
            fig, ax1 = self.fluct_data.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],contour_kw=contour_kw)

            xlabel = self.Domain.create_label(r"$%s$"%plane[0])
            ylabel = self.Domain.create_label(r"$%s$"%plane[1])

            ax[i].axes.set_xlabel(xlabel)
            ax[i].axes.set_ylabel(ylabel)

            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')
            fig.tight_layout()

            cbar=fig.colorbar(ax1,ax=ax[i].axes)
            cbar.set_label(r"$%s^\prime$"%comp)
            
        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
            
    # def plot_streaks(self,comp,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='wall',colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        
    #     vals_list = check_list_vals(vals_list)
    #     PhyTime = self.check_PhyTime(PhyTime)
        
    #     if y_limit is not None:
    #         y_lim_int = indexing.ycoords_from_norm_coords(self.avg_data,[y_limit],mode=y_mode)[0][0]
    #     else:
    #         y_lim_int = None

    #     kwargs = update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
    #     fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        
    #     for i,val in enumerate(vals_list):
    #         if colors is not None:
    #             color = colors[i%len(colors)]
    #             surf_kw['facecolor'] = color
    #         fig, ax1 = self.fluctDF.plot_isosurface(comp,val,time=PhyTime,y_limit=y_lim_int,
    #                                         x_split_pair=x_split_pair,fig=fig,ax=ax,
    #                                         surf_kw=surf_kw)
    #         ax.axes.set_ylabel(r'$x/\delta$')
    #         ax.axes.set_xlabel(r'$z/\delta$')
    #         ax.axes.invert_xaxis()

    #     return fig, ax1

        
    # def plot_fluct3D_xz(self,comp,y_vals,y_mode='half-channel',PhyTime=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        
    #     y_vals = check_list_vals(y_vals)
    #     PhyTime = self.check_PhyTime(PhyTime)
    #     y_int_vals  = indexing.ycoords_from_norm_coords(self.avg_data,y_vals,mode=y_mode)[0]

    #     axes_output = True if isinstance(ax,mpl.axes.Axes) else False
    #     kwargs = update_subplots_kw(kwargs,subplot_kw={'projection':'3d'},antialiased=True)
    #     fig, ax, axes_output = create_fig_ax_without_squeeze(len(y_int_vals),fig=fig,ax=ax,**kwargs)


    #     for i, val in enumerate(y_int_vals):
    #         fig, ax[i] = self.fluctDF.plot_surf(comp,'xz',val,time=PhyTime,x_split_pair=x_split_pair,fig=fig,ax=ax[i],surf_kw=surf_kw)
    #         ax[i].axes.set_ylabel(r'$x/\delta$')
    #         ax[i].axes.set_xlabel(r'$z/\delta$')
    #         ax[i].axes.set_zlabel(r'$%s^\prime$'%comp)
    #         ax[i].axes.invert_xaxis()

    #     if axes_output:
    #         return fig, ax[0]
    #     else:
    #         return fig, ax
        

    def plot_vector(self,plane,axis_vals,PhyTime=None,wall_units=True,spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        axis_vals = check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.fluct_data.CoordDF.check_plane(plane)

        if coord == 'y' and wall_units:
            int_vals = self._avg_data.get_coords_wall_units(coord,axis_vals,0)           
            axis_vals = self._avg_data.Wall_Coords(0).get_true_coords('y',axis_vals)
            title_symbol = get_symbol('wall_initial')

        else:
            int_vals = axis_vals = self.CoordDF.get_true_coords(coord,axis_vals)
            title_symbol = self.Domain.create_label(coord)

        x_size, z_size = self.fluct_data.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        for i, val in enumerate(int_vals):
            fig, ax[i] = self.fluct_data.plot_vector(plane,val,time=PhyTime,spacing=spacing,scaling=scaling,
                                                    fig=fig,ax=ax[i],quiver_kw=quiver_kw)

            xlabel = self.Domain.create_label(r"$%s$"%plane[0])
            ylabel = self.Domain.create_label(r"$%s$"%plane[1])

            ax[i].axes.set_xlabel(xlabel)
            ax[i].axes.set_ylabel(ylabel)

            ax[i].axes.set_title(r"$%s = %.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax[i].axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
       
    # @classmethod
    # def create_video(cls,axis_vals,comp,avg_data,contour=True,plane='xz',meta_data=None,path_to_folder='.',time_range=None,
    #                         abs_path=True,tgpost=False,x_split_list=None,plot_kw=None,lim_min=None,lim_max=None,
    #                         ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig=None,ax=None,**kwargs):

    #     axis_vals = misc_utils.check_list_vals(axis_vals)        
        
    #     if x_split_list is None:
    #         if meta_data is None:
    #             meta_data = cls._module._meta_class(path_to_folder,abs_path,tgpost=tgpost)
    #         x_coords = meta_data.CoordDF['x']
    #         x_split_list=[np.min(x_coords),np.max(x_coords)]

    #     if fig is None:
    #         if 'figsize' not in kwargs.keys():
    #             kwargs['figsize'] = [7*len(axis_vals),3*(len(x_split_list)-1)]
    #         fig = cplt.figure(**kwargs)
    #     if contour:
    #         plot_kw = cplt.update_pcolor_kw(plot_kw)
    #     def func(fig,time):
    #         axes = fig.axes
    #         for ax in axes:
    #             ax.remove()

    #         fluct_data = cls(time,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            
    #         if contour:
    #             fig, ax = fluct_data.plot_contour(comp,axis_vals,plane=plane,PhyTime=time,x_split_list=x_split_list,fig=fig,pcolor_kw=plot_kw)
    #         else:
    #             fig,ax = fluct_data.plot_fluct3D_xz(axis_vals,comp,time,x_split_list,fig,**plot_kw)
    #         ax[0].axes.set_title(r"$t^*=%.3g$"%time,loc='left')
    #         if fluct_func is not None:
    #             fluct_func(fig,ax,time,*fluct_args,**fluct_kw)
    #         if ax_func is not None:
    #             ax = ax_func(ax)
    #         for im in ax:
    #             im.set_clim(vmin=lim_min)
    #             im.set_clim(vmax=lim_max)

    #         fig.tight_layout()
    #         return ax

    #     return cplt.create_general_video(fig,path_to_folder,
    #                                     abs_path,func,time_range=time_range)
        
    def __str__(self):
        return self.fluct_data.__str__()

class x3d_fluct_z(_fluct_base):
    def _fluct_extract(self,time_inst_data_list,avg_data=None,path_to_folder='.',*args,**kwargs):
                
        if not isinstance(time_inst_data_list,(list,tuple)):
            time_inst_data_list = [time_inst_data_list]
            
        for i, time_inst_data in enumerate(time_inst_data_list):
            if isinstance(time_inst_data,self._module._inst_z_class):
                if i == 0:
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if i == 0:
                    inst_data = self._module._inst_z_class(time_inst_data,path_to_folder=path_to_folder,avg_data=avg_data,*args,**kwargs)
                else:
                    inst_data += self._module._inst_z_class(time_inst_data,path_to_folder=path_to_folder,avg_data=avg_data,*args,**kwargs)
        
        self.avg_data = inst_data._avg_data

        self.fluct_data = self._fluct_data_calc(inst_data,self.avg_data)

    def _hdf_extract(self, filename,key=None):
        if key is None:
            key= self.__class__.__name__

        self.avg_data = self._module._avg_z_class.from_hdf(filename,key=key+"/avg_data")
        self._meta_data = self._module._meta_class.from_hdf(filename,key=key+"/meta_data")
        self.fluct_data = fp.FlowStruct3D.from_hdf(filename,key=key+'/fluct_data')

    def _fluct_data_calc(self, inst_data, avg_data):
        
        avg_time = list(set([x[0] for x in avg_data.mean_data.index]))
        
        assert len(avg_time) == 1, "In this context therecan only be one time in avg_data"
        fluct = np.zeros((len(inst_data.inst_data.index),*inst_data.shape[:]),dtype=fp.rcParams['dtype'])
        j=0
        
        for j, (time, comp) in enumerate(inst_data.inst_data.index):
            avg_values = avg_data.mean_data[avg_time[0],comp]
            inst_values = inst_data.inst_data[time,comp]

            fluct[j] = inst_values - avg_values
        return fp.FlowStruct3D(inst_data.inst_data._coorddata,
                               fluct,
                               index=inst_data.inst_data.index)