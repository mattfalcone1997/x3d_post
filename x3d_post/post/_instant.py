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
from numbers import Number

_avg_z_class = x3d_avg_z
_avg_xz_class =x3d_avg_xz
_avg_xzt_class = x3d_avg_xzt

from ._meta import meta_x3d
_meta_class=meta_x3d

logger = logging.getLogger(__name__)

@njit(cache=True)
def _roots_cubic(a: Number, b: Number, c: Number, d: Number):
    delta0 = np.complex128(b*b - 3.*a*c)
    delta1 = np.complex128(2.*b*b*b -9*a*b*c + 27*a*a*d)

    C = (0.5*(delta1 + np.sqrt(delta1*delta1 - 4*delta0**3)))**(1/3)
    
    if abs(C) < 1e-8:
        C = 0.5*(delta1 - np.sqrt(delta1*delta1 - 4*delta0**3))
    
    
    cbr1 = 0.5*complex(-1.,3**(0.5))
    cbr2 = cbr1*cbr1
    
    if abs(delta0) < 1e-8 and abs(delta1)<1e-8:
        delta_divC1 = 0.
        delta_divC2 = 0.
        delta_divC3 = 0.
    else:
        delta_divC1 = delta0/C
        delta_divC2 = delta0/(C*cbr1)
        delta_divC3 = delta0/(C*cbr2)
        
    x1 = -(b + C      + delta_divC1) / (3.*a)
    x2 = -(b + cbr1*C + delta_divC2) / (3.*a)
    x3 = -(b + cbr2*C + delta_divC3) / (3.*a)
    
    return x1, x2, x3

@njit(cache=True)
def _det_calc33(A: np.ndarray):
    a = A[0,0]*(A[2,2]*A[1,1] - A[1,2]*A[2,1])
    b = - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0])
    c = A[0,2]*(A[1,0]*A[2,1] - A[2,0]*A[1,1])
    return a + b + c

@njit(cache=True)
def _matmul33(A: np.ndarray, B: np.ndarray):
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat[j,0] += A[j,i]*B[i,0]
            mat[j,1] += A[j,i]*B[i,1]
            mat[j,2] += A[j,i]*B[i,2]
                
    return mat

@njit(parallel=True, cache=True)
def _lambda_ci_int(A: np.ndarray):
    lambda_ci = np.zeros(A.shape[0])
    for i in prange(A.shape[0]):
        R = -_det_calc33(A[i])
        DD = _matmul33(A[i], A[i])
        Q = -0.5*(DD[0,0] + DD[1,1] + DD[2,2])

        l1, l2, l3 = _roots_cubic(1,0,Q, R)

        lambda_ci[i] = 0.5*(abs(l1.imag) + abs(l2.imag) + abs(l3.imag))
    return lambda_ci

@njit(cache=True)
def _get_eig2_33(A):

    p1 = A[0,1]*A[0,1] + A[0,2]*A[0,2] + A[1,2]*A[1,2] 
    q = (A[0,0] + A[1,1] + A[2,2])/3.
    
    p2 = (A[0,0]-q)*(A[0,0]-q) + (A[1,1]-q)*(A[1,1]-q) + (A[2,2]-q)*(A[2,2]-q)

    p = np.sqrt((p2 + 2.*p1)/6.)

    u_mat = A[:,:]/p

    u_mat[0,0] -= q/p
    u_mat[1,1] -= q/p
    u_mat[2,2] -= q/p

    det_b = _det_calc33(u_mat)

    phi= np.arccos(0.5*det_b)/3
    eig1 = q + 2*p*np.cos(phi)
    eig3 = q + 2*p*np.cos(phi + 2*np.pi/3)
    return 3*q - eig1 - eig3
    
@njit(parallel=True, cache=True)
def _lambda2_core(velo_grad):
    eig2 = np.zeros(velo_grad.shape[0])
    for i in prange(velo_grad.shape[0]):
        S = np.zeros((3,3))
        O = np.zeros((3,3))
        for j in range(3):
            for k in range(3):
                S[j,k] = 0.5*(velo_grad[i,j,k] + velo_grad[i,k,j]) 
                O[j,k] = 0.5*(velo_grad[i,j,k] - velo_grad[i,k,j]) 

        S2O2 = _matmul33(S,S) + _matmul33(O,O)
        eig2[i] = _get_eig2_33(S2O2)

    return eig2

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

        p1, p2 = plane
        c1 = chr(ord(p1)-ord('x') + ord('u'))
        c2 = chr(ord(p2)-ord('x') + ord('u'))
        for i, val in enumerate(int_vals):
            fig, ax[i] = self.inst_data.plot_vector((c1,c2),plane,val,time=PhyTime,spacing=spacing,scaling=scaling,
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
        
        
        shape1 = (*self.shape,3,3)
        velo_grad = np.zeros(shape1)
        
        for i, u in enumerate(velo_list):
            velo_field = self.inst_data[PhyTime,u]
            for j, x in enumerate(coord_list):

                velo_grad[:,:,:,i, j] = fp.Grad_calc(self.inst_data.CoordDF,velo_field,x)
        
        shape2 = (np.prod(self.inst_data.shape),3,3)
        velo_grad = velo_grad.reshape(shape2)
        
        lambda2 = _lambda2_core(velo_grad).reshape(self.shape)
        
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

    def lambda_ci_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
                
        D = np.zeros((*self.shape,3,3))

        for i,velo in enumerate(velo_list):
            velo_field = self.inst_data[PhyTime,velo]
            for j,coord in enumerate(coord_list):
                D[:,:,:,i,j] = fp.Grad_calc(self.inst_data.CoordDF,velo_field,coord)

        del velo_field

        size = np.prod(D.shape[:3])
        shape = (size,3, 3)

        lambda_ci =  _lambda_ci_int(D.reshape(shape)).reshape(self.inst_data.shape)

        return fp.FlowStruct3D(self.inst_data._coorddata,
                               {(PhyTime,'lambda_ci') : lambda_ci})



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