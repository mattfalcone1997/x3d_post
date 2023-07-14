from ._common import CommonData, classproperty
from abc import ABC, abstractproperty, abstractmethod
import flowpy as fp
from flowpy._api import copy_fromattr, sub
from flowpy.utils import check_list_vals
from flowpy.gradient import Curl
from ._data_handlers import inst_reader
from flowpy.plotting import (update_subplots_kw,
                            create_fig_ax_without_squeeze,
                            create_fig_ax_with_squeeze)
from ..style import get_symbol   
from ..utils import max_iteration                         
import numpy as np
import itertools                
import warnings            
import matplotlib as mpl          
import logging
from ._average import x3d_avg_z, x3d_avg_xz, x3d_avg_xzt
from numba import njit, prange
_avg_z_class = x3d_avg_z
_avg_xz_class =x3d_avg_xz
_avg_xzt_class = x3d_avg_xzt

from ._meta import meta_x3d
_meta_class=meta_x3d

try:
    import cupy as cnpy
    _cupy_avail = True
except:
    pass

logger = logging.getLogger(__name__)

class _Inst_base(CommonData,inst_reader,ABC):
    """
    ## CHAPSim_Inst
    This is a module for processing and visualising instantaneous data from CHAPSim
    """    
    _update_time = True
    
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        if not fromfile:
            self._inst_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)
    
    @classmethod
    def update_time(cls,val):
        cls._update_time = val
    
    @property
    def _coorddata(self):
        return self.inst_data._coorddata

    @property
    def Domain(self):
        return self.inst_data.Domain

    @property
    def times(self):
        return self.inst_data.times
    
    @classmethod
    def update_default_comps(cls,comps):
        if not all(isinstance(x, str) for x in comps):
            raise TypeError("All components must be characters")

        if not all(len(x) == 1 for x in comps):
            raise ValueError("All components must be length 1")
        
        cls._default_comps = tuple(comps)
        
    @abstractproperty
    def _avg_class(self):
        pass
    

    def _inst_extract(self,it,path='.',avg_data=None, comps=None, it0=None):
        """
        Instantiates CHAPSim_Inst by extracting data from the 
        CHAPSim rawdata results folder

        Parameters
        ----------
        time : int, float, or list
            Physical times to extract from results folder
        meta_data : CHAPSim_meta, optional
            a metadata instance, if not provided, it will be extracted, by default None
        path_to_folder : str, optional
            Path to the results folder, by default '.'
        abs_path : bool, optional
            Whether the path provided is an absolute path, by default True
        tgpost : bool, optional
            Whether the turbulence generator or spatially developing 
            region are processed, by default False

        """

        
        self._meta_data = self._module._meta_class(path)

        if comps is None:
            self._comps = list(self._default_comps)
        else:
            self._comps = list(comps)


        self.inst_data = self._extract_inst_xdmf(it,path,self._comps)

        self._avg_data = self._create_avg_data(it,path,it0,avg_data=avg_data)

        self._update_instant()
        
    def _update_instant(self):
        pass
    
    @abstractmethod
    def _create_avg_data(self,path,it0):
        pass

    @classmethod
    def from_hdf(cls,file_name,key=None):
        """
        Creates an instance of CHAPSim_inst by extracting an existing 
        saved instance from hdf file

        Parameters
        ----------
        file_name : str
            File path to existing hdf5 file
        key : str, optional
            path-like, hdf5 key to access the data within the file,
             by default None (class name)

        """
        return cls(file_name,fromfile=True,key=key)
    
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = fp.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.inst_data = fp.FlowStruct3D.from_hdf(file_name,key=key+'/inst_data')
        self._avg_data = self._avg_class.from_hdf(file_name,key=key+'/avg_data')
        
    @property
    def shape(self):
        return self.inst_data.shape

    def save_hdf(self,file_name,write_mode,key=None):
        """
        Saves the instance of the class to hdf5 file

        Parameters
        ----------
        file_name : str
            File path to existing hdf5 file
        write_mode : str
            The write mode for example append "a" or "w" see documentation for 
            h5py.File
        key : str, optional
            path-like, hdf5 key to access the data within the file,
             by default None (class name)
        """

        if key is None:
            key = self.__class__.__name__
        
        hdf_obj = fp.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(file_name,'a',key=key+'/meta_data')
        self.inst_data.to_hdf(file_name,key=key+'/inst_data',mode='a')
        self._avg_data.save_hdf(file_name,'a',key=key+'/avg_data')

    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = f"PhyTime provided ({PhyTime}) is not in the {self.__class__.__name__} datastruct, recovery impossible"
        
        err = ValueError(err_msg)
        warn = UserWarning(warn_msg)
        return self.inst_data.check_times(PhyTime,err,warn_msg)

    @sub
    def plot_contour(self,comp,axis_vals,plane='xz',wall_units=True,PhyTime=None,y_mode='wall',fig=None,ax=None,contour_kw=None,**kwargs):
        """
        Plot a contour along a given plane at different locations in the third axis

        Parameters
        ----------
        comp : str
            Component of the instantaneous data to be extracted e.g. "u" for the
            streamwise velocity

        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
             by default 'xz'
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        pcolor_kw : dict, optional
            Arguments passed to the pcolormesh function, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        axis_vals = check_list_vals(axis_vals)

        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.inst_data.CoordDF.check_plane(plane)

        if coord == 'y' and wall_units:
            int_vals = self._avg_data.get_coords_wall_units(coord,axis_vals,0)           
            axis_vals = self._avg_data.Wall_Coords(0).get_true_coords('y',axis_vals)
            title_symbol = get_symbol('wall_initial')

        else:
            int_vals = axis_vals = self.CoordDF.get_true_coords(coord,axis_vals)
            title_symbol = self.Domain.create_label(coord)

        x_size, z_size = self.inst_data.get_unit_figsize(plane)
        figsize=[x_size*len(axis_vals),z_size]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        for i,val in enumerate(int_vals):
            fig, ax1 = self.inst_data.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],contour_kw=contour_kw)

            xlabel = self.Domain.create_label(r"$%s$"%plane[0])
            ylabel = self.Domain.create_label(r"$%s$"%plane[1])

            ax[i].axes.set_xlabel(xlabel)
            ax[i].axes.set_ylabel(ylabel)

            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax

    @sub
    def plot_vector(self,plane,axis_vals,PhyTime=None,wall_units=True,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        """
        Create vector plot of a plane of the instantaneous flow

        Parameters
        ----------
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
            by default 'xz'
        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        spacing : tuple, optional
            [description], by default (1,1)
        scaling : int, optional
            [description], by default 1
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        quiver_kw : dict, optional
            Argument passed to matplotlib quiver plot, by default None
        
        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        axis_vals = check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.inst_data.CoordDF.check_plane(plane)

        if coord == 'y' and wall_units:
            int_vals = self._avg_data.get_coords_wall_units(coord,axis_vals,0)           
            axis_vals = self._avg_data.Wall_Coords(0).get_true_coords('y',axis_vals)
            title_symbol = get_symbol('wall_initial')

        else:
            int_vals = axis_vals = self.CoordDF.get_true_coords(coord,axis_vals)
            title_symbol = self.Domain.create_label(coord)

        x_size, z_size = self.inst_data.get_unit_figsize(plane)
        figsize=[x_size*len(axis_vals),z_size]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        for i, val in enumerate(int_vals):
            fig, ax[i] = self.inst_data.plot_vector(plane,val,time=PhyTime,spacing=spacing,scaling=scaling,
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
    
    @sub
    def lambda2_calc(self,PhyTime=None):
        """
        Calculation of lambda to visualise vortex cores

        Parameters
        ----------
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        x_start_index : int, optional
            streamwise location to start to calculation, by default None
        x_end_index : int, optional
            streamwise location to end to calculation, by default None
        y_index : int, optional
            First y_index of the mesh to be calculated, by default None

        Returns
        -------
        %(ndarray)s
            Array of the lambda2 calculation
        """
        
        PhyTime = self.check_PhyTime(PhyTime)

        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
        
        
        velo_grad = np.zeros((*self.shape,9))
        comp_iter = itertools.product(velo_list,coord_list)
        
        for i, (u, x) in enumerate(comp_iter):
            velo_field = self.inst_data[PhyTime,u]

            velo_grad[:,:,:,i] = fp.Grad_calc(self.inst_data.CoordDF,velo_field,x)
        
        velo_grad = velo_grad.reshape((*self.inst_data.shape,3,3))
        velo_grad_T = np.einsum('ijklm -> ijkml',velo_grad)
        strain_rate = 0.5*( velo_grad + velo_grad_T)
        
        rot_rate = 0.5*(velo_grad - velo_grad_T)

        del velo_field; del velo_grad; del velo_grad_T
        
        S2_Omega2 = np.matmul(strain_rate,strain_rate) + np.matmul(rot_rate,rot_rate)
        del strain_rate ; del rot_rate

        if fp.rcParams['use_cupy'] and _cupy_avail:
            S2_Omega2_eigvals, e_vecs = cnpy.linalg.eigh(S2_Omega2)
            del e_vecs; del S2_Omega2
            lambda2 = cnpy.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1].get()
            del S2_Omega2_eigvals
        else:
            lambda2 = _compute_eig2(S2_Omega2)        
        
        return fp.FlowStruct3D(self.inst_data._coorddata,{(PhyTime,'lambda_2'):lambda2})

    @sub
    def plot_lambda2(self,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='half_channel',Y_plus=True,colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        """
        Creates isosurfaces for the lambda 2 criterion

        Parameters
        ----------
        vals_list : list of floats
            isovalues to be plotted
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        ylim : float, optional
            wall-normal extent of the isosurface plot, by default None
        Y_plus : bool, optional
            Whether the above value is in wall units, by default True
        avg_data : CHAPSim_AVG, optional
            Instance of avg_data need if Y_plus is True, by default None
        colors : list of str, optional
            list to represent the order of the colors to be plotted, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects

        """
        PhyTime = self.check_PhyTime(PhyTime)
        vals_list = check_list_vals(vals_list)

        if y_limit is not None:
            y_lim_int = self._avg_data.ycoords_from_norm_coords([y_limit],inst_time=PhyTime,mode=y_mode)[0][0]
        else:
            y_lim_int = None

        kwargs = update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        lambda_2DF = self.lambda2_calc(PhyTime)
        for i,val in enumerate(vals_list):
            if colors is not None:
                color = colors[i%len(colors)]
                surf_kw['facecolor'] = color
            fig, ax1 = lambda_2DF.plot_isosurface('lambda_2',val,time=PhyTime,y_limit=y_lim_int,
                                            x_split_pair=x_split_pair,fig=fig,ax=ax,
                                            surf_kw=surf_kw)
            ax.axes.set_ylabel(r'$x/\delta$')
            ax.axes.set_xlabel(r'$z/\delta$')
            ax.axes.invert_xaxis()

        return fig, ax1

    def Q_crit_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
                
        D = np.zeros((*self.shape,3,3))

        for i,velo in enumerate(velo_list):
            velo_field = self.inst_data[PhyTime,velo]
            for j,coord in enumerate(coord_list):
                D[:,:,:,i,j] = fp.Grad_calc(self.CoordDF,velo_field,coord)

        del velo_field

        Q = 0.5*(np.trace(D,axis1=3,axis2=4,dtype=fp.rcParams['dtype'])**2 - \
            np.trace(np.matmul(D,D,dtype=fp.rcParams['dtype']),axis1=3,axis2=4,dtype=fp.rcParams['dtype']))
        del D
        return fp.flowstruct3D(self._coorddata,{(PhyTime,'Q'):Q})

    def plot_Q_criterion(self,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='half_channel',colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        vals_list = check_list_vals(vals_list)

        if y_limit is not None:
            y_lim_int = self._avg_data.ycoords_from_norm_coords([y_limit],inst_time=PhyTime,mode=y_mode)[0][0]
        else:
            y_lim_int = None

        kwargs = update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        Q = self.Q_crit_calc(PhyTime)
        for i,val in enumerate(vals_list):
            if colors is not None:
                color = colors[i%len(colors)]
                surf_kw['facecolor'] = color
            fig, ax1 = Q.plot_isosurface('Q',val,time=PhyTime,y_limit=y_lim_int,
                                            x_split_pair=x_split_pair,fig=fig,ax=ax,
                                            surf_kw=surf_kw)
            ax.axes.set_ylabel(r'$x/\delta$')
            ax.axes.set_xlabel(r'$z/\delta$')
            ax.axes.invert_xaxis()

        return fig, ax1

    def vorticity_calc(self,PhyTime=None):
        """
        Calculate the vorticity vector

        Parameters
        ----------
        PhyTime : float, optional
            Physical time, by default None

        Returns
        -------
        datastruct
            Datastruct with the vorticity vector in it
        """

        PhyTime = self.check_PhyTime(PhyTime)

        vorticity = np.zeros((3,*self.inst_data.shape),dtype='f8')
        velo_vector = self.inst_data.values[:3]

        vorticity = Curl(self.inst_data.CoordDF,
                                 velo_vector,
                                 polar=self.Domain.is_polar)

        index = [(PhyTime,x) for x in ['x','y','z']]
        return fp.FlowStruct3D(self.inst_data._coorddata,vorticity,index=index)

    @sub
    def plot_vorticity_contour(self,comp,plane,axis_vals,PhyTime=None,x_split_list=None,y_mode='half_channel',pcolor_kw=None,fig=None,ax=None,**kwargs):
        """
        Creates a contour plot of the vorticity contour

        Parameters
        ----------
        comp : str
            Component of the vorticity to be extracted e.g. "x" for 
            \omega_z, the spanwise vorticity
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
            by default 'xz'
        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        VorticityDF = self.vorticity_calc(PhyTime=PhyTime)

        plane = self.Domain.out_to_in(plane)
        axis_vals = check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = VorticityDF.CoordDF.check_plane(plane)

        if coord == 'y' and self._avg_available:
            axis_vals = self._avg_data.ycoords_from_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
            int_vals = self._avg_data.ycoords_from_norm_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
        else:
            int_vals = axis_vals = self.CoordDF.get_true_coords(coord,axis_vals)
            y_mode = 'half-channel'
            
        x_size, z_size = VorticityDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = get_symbol(coord,y_mode,False)

        for i,val in enumerate(int_vals):
            fig, ax1 = VorticityDF.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],pcolor_kw=pcolor_kw)
            ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
            ax1.axes.set_ylabel(r"$%s/\delta$" % plane[1])
            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_entrophy(self):
        pass
    def __str__(self):
        return self.inst_data.__str__()
    def __iadd__(self,inst_data):
        assert self.CoordDF.equals(inst_data.CoordDF), "CHAPSim_Inst are not from the same case"
        assert self.NCL == inst_data.NCL, "CHAPSim_Inst are not from the same case"

        self.inst_data.concat(inst_data.inst_data)
        return self
    
@njit(parallel=True)
def _compute_eig2(array: np.ndarray):
    eig2 = np.zeros(array.shape[:3])
    for i in prange(array.shape[0]):
        for j in prange(array.shape[1]):
            for k in prange(array.shape[2]):
                p1 = array[i,j,k,0,0]*array[i,j,k,0,0] \
                    + array[i,j,k,1,1]*array[i,j,k,1,1] \
                    + array[i,j,k,2,2]*array[i,j,k,2,2] 
    
                p1 = array[i,j,k,0,1]*array[i,j,k,0,1] \
                    + array[i,j,k,0,2]*array[i,j,k,0,2] \
                    + array[i,j,k,1,2]*array[i,j,k,1,2] 

                q = (array[i,j,k,0,0] + array[i,j,k,1,1] + array[i,j,k,2,2])/3.
                
                p2 = (array[i,j,k,0,0]-q)*(array[i,j,k,0,0]-q) \
                    + (array[i,j,k,1,1]-q)*(array[i,j,k,1,1]-q) \
                    + (array[i,j,k,2,2]-q)*(array[i,j,k,2,2]-q)
    
                p = np.sqrt((p2 + 2.*p1)/6.)

                u_mat = array[i,j,k,:,:]/p

                u_mat[0,0] -= q/p
                u_mat[1,1] -= q/p
                u_mat[2,2] -= q/p

                det_b = u_mat[0,0]*(u_mat[1,1]*u_mat[2,2] \
                                            -u_mat[1,2]*u_mat[2,1]) \
                        - u_mat[0,1]*(u_mat[1,0]*u_mat[2,2] \
                                            -u_mat[1,2]*u_mat[2,0]) \
                        + u_mat[0,2]*(u_mat[1,0]*u_mat[2,1] \
                                            -u_mat[1,1]*u_mat[2,0])
                phi= np.arccos(0.5*det_b)/3 + 4.*np.pi/3.
                eig2[i,j,k] = q + 2.*p*np.cos(phi)
    return eig2

class x3d_inst_z(_Inst_base):
    
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_z_class
    
    def _create_avg_data(self,it,path,it0,avg_data=None):
        it = max_iteration(path)
        if avg_data is not None :
            return avg_data
        else:
            if self._module._avg_z_class.avg_avail(it,path):
                self._avg_available = True
                return self._module._avg_z_class(it,path,
                                                 it0=it0)
                
            else:
                warnings.warn("No average available")
                self._avg_available = False
                
class x3d_inst_xz(_Inst_base):
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_z_class
    
    def _create_avg_data(self,it,path,it0,avg_data=None):
        it = max_iteration(path)
        if avg_data is not None :
            return avg_data
        else:
            if self._module._avg_xz_class.avg_avail(it,path):
                self._avg_available = True
                return self._module._avg_xz_class(it,path,
                                                 it0=it0)
                
            else:
                warnings.warn("No average available")
                self._avg_available = False

class x3d_inst_xzt(_Inst_base):
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_xzt_class                    
    
    def _create_avg_data(self,it,path,it0,avg_data=None):
        if avg_data is not None :
            return avg_data
        else:
            if self._avg_class.avg_avail(it,path):
                self._avg_available = True
                return self._avg_class(path,its=[it])
                
            else:
                warnings.warn("No average available")
                self._avg_available = False