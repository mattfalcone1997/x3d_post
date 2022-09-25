import warnings
import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import os
import gc
import itertools
import copy
from functools import partial
from abc import ABC, abstractmethod, abstractproperty

from ._meta import meta_x3d
from ._common import CommonData
import flowpy as fp
from ._data_handlers import (stat_z_handler,
                             stat_xz_handler)
from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_line_kw,
                             )
from flowpy.utils import (check_list_vals)

_meta_class = meta_x3d

class _AVG_base(CommonData,ABC):
    def __init__(self,*args,from_hdf=False,**kwargs):

        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_avg(*args,**kwargs)

    @abstractmethod
    def _extract_avg(self,*args,**kwargs):
        raise NotImplementedError        
    
    @abstractmethod
    def Wall_Coords(self,*args,**kwargs):
        pass

    @property
    def shape(self):
        return self.mean_data.shape

    @property
    def Domain(self):
        return self.mean_data.Domain

    @property
    def _coorddata(self):
        return self.mean_data._coorddata
        
    @property
    def times(self):
        return np.array(self.mean_data.times)

    @abstractmethod
    def get_coords_wall_units(self):
        pass

    def get_times(self):
        return ["%.9g"%x for x in self.times]

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(from_hdf=True,*args,**kwargs)

    def save_hdf(self,fn, mode,key=None):
        """
        Saving the CHAPSim_AVG classes to a file in hdf5 file format

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        key = self._get_hdf_key(key)

        hdf_obj = fp.hdfHandler(fn,mode=mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(fn,'a',key=key+'/meta_data')
        self.mean_data.to_hdf(fn,'a',key=key+'/mean_data',mode='a')
        self.uu_data.to_hdf(fn,'a',key=key+'/uu_data',mode='a')

        return hdf_obj

    @abstractmethod
    def _hdf_extract(self,*args,**kwargs):
        pass

    def check_PhyTime(self,PhyTime):
        """
        Checks whether the physical time provided is valid
        and if not whether it can be recovered.

        Parameters
        ----------
        PhyTime : float or int
            Input Physical time to be checked

        Returns
        -------
        float or int
            Correct or corrected physical time
        """

        warn_msg = f"PhyTime invalid ({PhyTime}), variable being set to only PhyTime present in datastruct"
        err_msg = KeyError("PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible")
        
        return self.mean_data.check_outer(PhyTime,err_msg,warn_msg) 

class _AVG_developing(_AVG_base):
    @abstractmethod
    def _return_index(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _return_xaxis(self,*args,**kwargs):
        raise NotImplementedError

    def _wall_unit_calc(self,PhyTime):
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        re = self.metaDF['re']
        
        tau_w = self._tau_calc(PhyTime)
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(re)
        delta_v_star = (nu_star/u_tau_star)/re

        return u_tau_star, delta_v_star

    def _y_plus_calc(self,PhyTime):

        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(self._shape[1], self.NCL[1] // 2)


        y_coord = self.CoordDF['y'][np.newaxis,:y_plus_shape[1]]
        return delta_v_star*(1-abs(y_coord))

    def _int_thickness_calc(self,PhyTime):

        U0_index = 0 if self.Domain.is_polar else self.shape[0] // 2
        y_coords = self.CoordDF['y']

        U_mean = self.mean_data[PhyTime,'u'].copy()
        U0 = U_mean[U0_index,np.newaxis]

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = 1 - U_mean/U0

        mom_thickness = 0.5*integrate_simps(theta_integrand,y_coords,axis=0)
        disp_thickness = 0.5*integrate_simps(delta_integrand,y_coords,axis=0)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor

    def _velo_scale_calc(self,PhyTime):
            
        u_velo = self.mean_data[PhyTime,'u'].squeeze()
        ycoords = self.CoordDF['y']

        bulk_velo = 0.5*integrate_simps(u_velo,ycoords,axis=0)
            
        return bulk_velo

    def _tau_calc(self,PhyTime):
        
        u_velo = self.mean_data[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        mu_star = 1.0
        tau_star = -mu_star*(u_velo[-1,:] - u_velo[-2,:])/(ycoords[-1]-ycoords[-2])        

        return tau_star

    def _Cf_calc(self,PhyTime):
        rho_star = 1.0
        re = self.metaDF['re']
        tau_star = self.tau_calc(PhyTime)
        bulk_velo = self.velo_scale_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/re)*tau_star

        return skin_friction

    def _eddy_visc_calc(self,PhyTime):
        uv = self.uu_data[PhyTime,'uv']
        U_mean = self.mean_data[PhyTime,'u']
        V_mean = self.mean_data[PhyTime,'v']

        dUdy = fp.Grad_calc(self.CoordDF,U_mean,'y')
        dVdx = fp.Grad_calc(self.CoordDF,V_mean,'x')

        re = self.metaDF['re']

        mu_t = -uv*re/(dUdy + dVdx)
        return mu_t                


class x3d_avg_z(_AVG_developing,stat_z_handler):
    def _return_index(self,x_val):
        return self.CoordDF.index_calc('x',x_val)

    def _return_xaxis(self):
            return self.CoordDF['x']

    def _extract_avg(self,it,path,it0=None):
        
        self._meta_data = self._module._meta_class(path)

        self.mean_data = self._extract_umean(path,it,it0)
        self.uu_data = self._extract_uumean(path,it,it0)

    def _hdf_extract(self, fn,key=None):
        key = self._get_hdf_key(key)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')

        self.mean_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/mean_data')
        self.uu_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/uu_data')

        return fp.hdfHandler(fn,'r',key=key)

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = self._meta_data.save_hdf(fn,mode,key=key+'/meta_data')
        self.mean_data.to_hdf(fn,key=key+'/mean_data')
        self.uu_data.to_hdf(fn,key=key+'/uu_data')

        return h5_obj

    def Translate(self,translation):
                
        self.mean_data.Translate(translation)
        self.uu_data.Translate(translation)

        self.CoordDF.Translate([*translation[::-1],0])
    
    def Wall_Coords(self,axis_val,PhyTime=None):
        _, delta_v = self.wall_unit_calc(PhyTime)
        index = self.CoordDF.index_calc('x',axis_val)[0]
        return  self.CoordDF / delta_v[index]

    def get_coords_wall_units(self,comp,coords,axis_val,PhyTime=None):
        indices = self.Wall_Coords(axis_val,PhyTime).index_calc(comp,coords)
        return self.CoordDF[comp][indices]


    def int_thickness_calc(self, PhyTime=None):
        """
        Calculates the integral thicknesses and shape factor 

        Parameters
        ----------
        PhyTime : float or int, optional
            Physical ime, by default None

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        PhyTime = self.check_PhyTime(PhyTime)
        return self._int_thickness_calc(PhyTime)        

    def wall_unit_calc(self,PhyTime=None):
            
        PhyTime = self.check_PhyTime(PhyTime)
        return self._wall_unit_calc(PhyTime)

    def velo_scale_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)
        return self._velo_scale_calc(PhyTime)

    bulk_velo_calc = velo_scale_calc

    def tau_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)            
        return self._tau_calc(PhyTime)        

    def plot_shape_factor(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)

        _, _, shape_factor = self.int_thickness_calc(PhyTime)
        
        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']
        line_kw = update_line_kw(line_kw,label = r"$H$")
        ax.cplot(x_coords,shape_factor,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$H$")

        return fig, ax

    def plot_mom_thickness(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
    
        PhyTime = self.check_PhyTime(PhyTime)
        _, theta, _ = self.int_thickness_calc(PhyTime)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        x_coords = self.CoordDF['x']

        line_kw = update_line_kw(line_kw,label=r"$\theta$")
        ax.cplot(x_coords,theta,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$\theta$")
        fig.tight_layout()

        return fig, ax

    def plot_disp_thickness(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        delta, _, _ = self.int_thickness_calc(PhyTime)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']

        line_kw = update_line_kw(line_kw,label=r"$\delta^*$")
        ax.cplot(x_coords,delta,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$\delta^*$")
        fig.tight_layout()

        return fig, ax        


    def plot_mean_flow(self,comp,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        fig, ax = self.mean_data.plot_line(comp,'y',x_vals,
                            time=PhyTime,labels=labels,
                            fig=fig,ax=ax,line_kw=line_kw,**kwargs)


        x_label = self.Domain.create_label(r"$y$")
        y_label = self.Domain.create_label(r"$\bar{%s}$"%comp)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        return fig, ax

    def _get_uplus_yplus_transforms(self,PhyTime,x_val):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x',x_val)[0]
        

        x_transform = lambda y:  y/delta_v[x_index]
        y_transform = lambda u: u/u_tau[x_index]
        
        return x_transform, y_transform
    
    def plot_flow_wall_units(self,x_vals,plot_uplus_yplus=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)

        for x in x_vals:
            label = self.Domain.create_label(r"$x = %.3g$"%x)
            x_transform, y_transform = self._get_uplus_yplus_transforms(PhyTime, x)
            fig, ax = self.mean_data.plot_line('u','y',
                                                x,
                                                transform_xdata = x_transform,
                                                transform_ydata = y_transform,
                                                labels=[label],
                                                channel_half=True,
                                                time=PhyTime,
                                                fig=fig,
                                                ax=ax,
                                                line_kw=line_kw,
                                                **kwargs)

        labels = [l.get_label() for l in ax.get_lines()]
        if not r"$\bar{u}^+=y^+$" in labels and plot_uplus_yplus:
            uplus_max = np.amax([l.get_ydata() for l in ax.get_lines()])
            u_plus_array = np.linspace(0,uplus_max,100)

            ax.cplot(u_plus_array,u_plus_array,label=r"$\bar{u}^+=y^+$",color='r',linestyle='--')

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')

        return fig, ax

    def plot_Reynolds(self,comp,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            raise ValueError("Reynolds stress component %s not found"%comp) 

        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        if comp == 'uv':
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        fig, ax = self.uu_data.plot_line(comp,'y',x_vals,time=PhyTime,labels=labels,fig=fig,channel_half=True,
                                    transform_ydata=transform_y, ax=ax,line_kw=line_kw,**kwargs)

        sign = "-" if comp == 'uv' else ""

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = self.Domain.avgStyle(uu_label)
        y_label = r"$%s %s$"%(sign,avg_label)

        x_label = self.Domain.create_label(r"$y$")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        return fig, ax

    def plot_Reynolds_x(self,comp,y_vals_list,y_mode='half-channel',PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            raise ValueError("Reynolds stress component %s not found"%comp) 

        PhyTime = self.check_PhyTime(PhyTime)

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))

        line_kw = update_line_kw(line_kw)
        avg_label = self.Domain.avgStyle(uu_label)
        if y_vals_list == 'max':
            line_kw['label'] = r"$%s_{max}$"%avg_label
            
            fig, ax = self.uu_data.plot_line_max(comp,'x',time=PhyTime,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        else:
            msg = "This method needs to be reimplemented only max can currently be used"
            raise NotImplementedError(msg)

        ax.set_ylabel(r"$%s$"%avg_label)

        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_bulk_velocity(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        bulk_velo = self.bulk_velo_calc(PhyTime)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax =create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']
        line_kw = update_line_kw(line_kw,label=r"$U_{b}$")

        ax.cplot(x_coords,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        
        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax
        
    def plot_skin_friction(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)
        skin_friction = self._Cf_calc(PhyTime)
        x_coords = self.CoordDF['x']

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = update_line_kw(line_kw,label=r"$C_f$")
        ax.cplot(x_coords,skin_friction,**line_kw)
        ax.set_ylabel(r"$C_f$")
        
        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_eddy_visc(self,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)
        
        PhyTime = self.check_PhyTime(PhyTime)

        mu_t = self._eddy_visc_calc(PhyTime)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        fig, ax = self.uu_data.plot_line_data(mu_t,'y',x_vals,labels=labels,
                                fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        ax.set_ylabel(r"$\mu_t/\mu_0$")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        ax.set_xlim([-1,-0.1])

        return fig, ax        


class x3d_avg_xz(_AVG_base,stat_xz_handler):
    def _extract_avg(self,it,path,it0=None):
        
        self._meta_data = self._module._meta_class(path)

        self.mean_data = self._extract_umean(path,it,it0)
        self.uu_data = self._extract_uumean(path,it,it0)

    def _hdf_extract(self, fn,key=None):
        key = self._get_hdf_key(key)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')

        self.mean_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/mean_data')
        self.uu_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/uu_data')

        return fp.hdfHandler(fn,'r',key=key)

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = self._meta_data.save_hdf(fn,mode,key=key+'/meta_data')
        self.mean_data.to_hdf(fn,key=key+'/mean_data')
        self.uu_data.to_hdf(fn,key=key+'/uu_data')

        return h5_obj

    def wall_unit_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star
        REN = self.metaDF['re']
        
        tau_w = self.tau_calc(PhyTime)
    
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def Wall_Coords(self,PhyTime=None):
        _, delta_v = self.wall_unit_calc(PhyTime)
        return  self.CoordDF / delta_v

    def get_coords_wall_units(self,comp,coords,PhyTime=None):
        indices = self.Wall_Coords(PhyTime).index_calc(comp,coords)
        return self.CoordDF[comp][indices]

    def int_thickness_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        index = self.shape // 2

        U_mean = self.mean_data[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']
        U0 = U_mean[index]

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = 1 - U_mean/U0

        mom_thickness = 0.5*integrate_simps(theta_integrand,y_coords)
        disp_thickness = 0.5*integrate_simps(delta_integrand,y_coords)

        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor

    def tau_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        u_velo = self.mean_data[PhyTime,'u']
        ycoords = self.CoordDF['y']

        mu_star = 1.0
        return mu_star*(u_velo[2]-u_velo[1])/(ycoords[2]-ycoords[1])

    def _velo_scale_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        u_velo = self.mean_data[PhyTime,'u']
        ycoords = self.CoordDF['y']

        bulk_velo = 0.5*integrate_simps(u_velo,ycoords)
        return bulk_velo

    bulk_velo_calc = _velo_scale_calc

    def Cf_calc(self,PhyTime):
        PhyTime = self.check_PhyTime(PhyTime)

        rho_star = 1.0
        REN = self.metaDF['re']
        tau_star = self.tau_calc(PhyTime)
        bulk_velo = self._velo_scale_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star

        return skin_friction

    def _eddy_visc_calc(self,PhyTime):
        uv = self.uu_data[PhyTime,'uv']

        U = self.mean_data[PhyTime,'u']

        dUdy = fp.Grad_calc(self.CoordDF,U,'y')
        re = self.metaDF['re']

        mu_t = -uv*re/dUdy
        return mu_t        

    def plot_mean_flow(self,comp,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
    
        PhyTime = self.check_PhyTime(PhyTime)

        fig, ax = self.mean_data.plot_line(comp,time=PhyTime,
                            fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        x_label = self.Domain.create_label(r"$y$")
        y_label = self.Domain.create_label(r"$\bar{u}$")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig, ax        

    def _get_uplus_yplus_transforms(self,PhyTime):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        
        x_transform = lambda y:  y/delta_v
        y_transform = lambda u: u/u_tau
        
        return x_transform, y_transform        

    def plot_flow_wall_units(self,PhyTime=None,fig=None,ax=None,line_kw=None,plot_sublayer=True,**kwargs):
    
        PhyTime = self.check_PhyTime(PhyTime)

        
        y_plus_trans, u_plus_trans = self._get_uplus_yplus_transforms(PhyTime)


        fig, ax = self.mean_data.plot_line('u',
                                            channel_half=True,
                                            time=PhyTime,
                                            transform_xdata=y_plus_trans,
                                            transform_ydata=u_plus_trans,
                                            fig=fig,ax=ax,
                                            line_kw=line_kw,
                                            **kwargs)

        if plot_sublayer:
            u = np.amax(self.mean_data[PhyTime,'u'])
            u_plus_array = np.linspace(0,u_plus_trans(u),100)
            ax.cplot(u_plus_array,u_plus_array,
                     label=r"$\bar{u}^+=y^+$",
                     color='r',
                     linestyle='--')

        

        ax.set_xscale('log')
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")

        return fig, ax        

    def plot_Reynolds(self,comp,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            msg = "Reynolds stress component %s not found"%comp
            raise ValueError(msg) 

        PhyTime = self.check_PhyTime(PhyTime)

        if comp == 'uv' and self.Domain.is_channel:
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        fig, ax = self.uu_data.plot_line(comp,time=PhyTime,fig=fig,channel_half=True,
                                    transform_ydata=transform_y,
                                    ax=ax,line_kw=line_kw,**kwargs)
    
        sign = "-" if comp == 'uv' else ""


        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = self.Domain.avgStyle(uu_label)
        y_label = r"$%s %s$"%(sign,avg_label)

        x_label = self.Domain.create_label(r"$y$")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # self.Domain.styleAxes(ax)

        return fig, ax        

    def plot_eddy_visc(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)

        mu_t = self._eddy_visc_calc(PhyTime)

        fig, ax = self.uu_data.plot_line_data(mu_t,fig=fig,ax=ax,
                                                  line_kw=line_kw,**kwargs)

        ax.set_ylabel(r"$\mu_t/\mu_0$")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        ax.set_xlim([-1,-0.1])
        return fig, ax        