from re import L
import warnings
import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import os

from abc import ABC, abstractmethod

from ._meta import meta_x3d
from ._common import (CommonData,
                      CommonTemporalData,)
from ..style import get_symbol
from ..utils import get_iterations
import flowpy as fp
from ._data_handlers import (stathandler_base,
                             stat_z_handler,
                             stat_xz_handler,
                             stat_xzt_handler)

from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_line_kw,
                             )
from flowpy.utils import (check_list_vals,)
from itertools import chain
_meta_class = meta_x3d

class _AVG_base(CommonData,stathandler_base,ABC):

    def __init__(self,*args,from_hdf=False,**kwargs):

        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_avg(*args,**kwargs)

    def _extract_avg(self,it,path,it0=None):
        
        self._meta_data = self._module._meta_class(path)

        self.mean_data = self._extract_umean(path,it,it0)
        self.uu_data = self._extract_uumean(path,it,it0)
        
        if self._has_data(path,'uuu_mean'):
            self.uuu_data = self._extract_uuumean(path,it,it0)
        
        if self._has_data(path,'uuuu_mean'):
            self.uuuu_data = self._extract_uuuumean(path,it,it0)
      
    def _has_data(self,path,comp):
        path = os.path.join(path,'statistics')
        files = os.listdir(path)
        return any(comp in f for f in files)
    
    @abstractmethod
    def Wall_Coords(self,*args,**kwargs):
        pass

    @property
    def shape(self):
        return self.mean_data.shape

    @property
    def Domain(self):
        return self._meta_data.Domain

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

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = self._meta_data.save_hdf(fn,mode,key=key+'/meta_data')
        self.mean_data.to_hdf(fn,key=key+'/mean_data')
        self.uu_data.to_hdf(fn,key=key+'/uu_data')

        if hasattr(self,'uuu_data'):
            self.uuu_data.to_hdf(fn,key=key+'/uuu_data')
            
        if hasattr(self,'uuuu_data'):
            self.uuuu_data.to_hdf(fn,key=key+'/uuuu_data')            
        
        return h5_obj

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


    def _extract_umean(self,path,it,it0):

        names = ['umean','vmean','wmean','pmean']
        l = self._get_data(path,'uvwp_mean',names,it,4)

        if it0 is not None:
            l0 = self._get_data(path,'uvwp_mean',names,it0,4)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)


        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['u','v','w','p']
        index = self._get_index(it,comps)
        
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        mean_data = self._flowstruct_class(coorddata,
                                            l,
                                            index=index)
        return mean_data

    def _extract_uumean(self,path,it,it0):
        names = ['uumean','vvmean','wwmean',
                 'uvmean','uwmean','vwmean']

        l = self._get_data(path,'uu_mean',names,it,6)
        if it0 is not None:
            l0 = self._get_data(path,'uu_mean',names,it0,6)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uu','vv','ww','uv','uw','vw']

        self._check_attr('mean_data')
        i =0
        for time in self.mean_data.times:
            for comp in comps:
                comp1, comp2 = comp
                u1 = self.mean_data[time,comp1]
                u2 = self.mean_data[time,comp2]

                l[i] = l[i] - u1*u2
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        uu_data =  self._flowstruct_class(coorddata,
                                          l,
                                          index=index)
        return uu_data
    
    def _extract_uuumean(self,path,it,it0):
        names = ['uuumean','uuvmean','uuwmean',
                 'uvvmean','uvwmean','uwwmean',
                 'vvvmean','vwwmean','vvwmean','wwwmean']

        l = self._get_data(path,'uuu_mean',names,it,10)
        if it0 is not None:
            l0 = self._get_data(path,'uuu_mean',names,it0,10)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)
        if hasattr(it,'__iter__'):
            items = list(chain(*[ np.array([0,6,8])+10*i for i in range(len(it))]))
            l = l[items]
        else:
            l = l[[0,6,8]]
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uuu','vvv','www']

        self._check_attr('mean_data')
        self._check_attr('uu_data')
        i = 0
        for time in self.mean_data.times:    
            for comp in comps:
                comp_uu_12 = comp[:2]
                comp_uu_13 = comp[0] + comp[2]
                comp_uu_23 = comp[1:]

                comp_u_1 = comp[0]
                comp_u_2 = comp[1]
                comp_u_3 = comp[2]

                u1 = self.mean_data[time,comp_u_1]
                u2 = self.mean_data[time,comp_u_2]
                u3 = self.mean_data[time,comp_u_3]

                u1u2 = self.uu_data[time,comp_uu_12]
                u1u3 = self.uu_data[time,comp_uu_13]
                u2u3 = self.uu_data[time,comp_uu_23]

                l[i] = l[i] - (u1*u2*u3 + u1*u2u3 \
                            + u2*u1u3 + u3*u1u2)
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        uuu_data =  self._flowstruct_class(coorddata,
                                           l,
                                           index=index)
        return uuu_data
    
    def _extract_uuuumean(self,path,it,it0):
        l = self._get_data(path,'uuuu_mean',None,it,3)
        if it0 is not None:
            l0 = self._get_data(path,'uuuu_mean',None,it0,3)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uuuu','vvvv','wwww']

        self._check_attr('mean_data')
        self._check_attr('uu_data')
        self._check_attr('uuu_data')

        i = 0
        for time in self.mean_data.times:    
            for comp in comps:
                comp_uuu = comp[:3]
                comp_uu = comp[:2]
                comp_u = comp[0]


                u = self.mean_data[time,comp_u]
                uu = self.uu_data[time,comp_uu]
                uuu = self.uuu_data[time,comp_uuu]

                u2 = uu + u*u
                u3 = uuu +3*u*u2 -2*u*u*u
                l[i] = l[i] - 4*u3*u + 6*u2*u*u - 3*u*u*u*u
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)


        uuu_data =  self._flowstruct_class(coorddata,
                                           l,
                                           index=index)
        return uuu_data
    
    
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

        dUdy = fp.Grad_calc(self.mean_data.CoordDF,U_mean,'y')
        dVdx = fp.Grad_calc(self.mean_data.CoordDF,V_mean,'x')

        re = self.metaDF['re']

        mu_t = -uv*re/(dUdy + dVdx)
        return mu_t                


class x3d_avg_z(_AVG_developing,stat_z_handler):
    def _return_index(self,x_val):
        return self.CoordDF.index_calc('x',x_val)

    def _return_xaxis(self):
            return self.CoordDF['x']

    def _hdf_extract(self, fn,key=None):
        key = self._get_hdf_key(key)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')

        self.mean_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/mean_data')
        self.uu_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/uu_data')

        h5_obj = fp.hdfHandler(fn,'r',key=key)
        if 'uuu_data' in h5_obj.keys():
            self.uuu_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/uuu_data')
        
        if 'uuuu_data' in h5_obj.keys():
            self.uuuu_data = fp.FlowStruct2D.from_hdf(fn,key=key+'/uuuu_data')                
        
        return h5_obj

    def Translate(self,translation):
                
        self.mean_data.Translate(translation)
        self.uu_data.Translate(translation)

        args = {'y':translation[0],'x':translation[1]}
        self.CoordDF.Translate(**args)
    
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
        

        x_transform = lambda y:  y[1:]/delta_v[x_index] if isinstance(y,np.ndarray) else y/delta_v[x_index]
        y_transform = lambda u: u[1:]/u_tau[x_index] if isinstance(u,np.ndarray) else u/u_tau[x_index]
        
        return x_transform, y_transform
    
    def plot_flow_wall_units(self,x_vals,plot_sublayer=True,PhyTime=None,fig=None,ax=None,line_kw:dict =None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)

        x_labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]
        for i, x in enumerate(x_vals):

            if line_kw is not None:
                labels = [line_kw.get('label',x_labels[i])]
            else:
                labels  = [x_labels[i]]
                            
            x_transform, y_transform = self._get_uplus_yplus_transforms(PhyTime, x)
            fig, ax = self.mean_data.plot_line('u','y',
                                                x,
                                                transform_xdata = x_transform,
                                                transform_ydata = y_transform,
                                                labels=labels,
                                                channel_half=True,
                                                time=PhyTime,
                                                fig=fig,
                                                ax=ax,
                                                line_kw=line_kw,
                                                **kwargs)

        labels = [l.get_label() for l in ax.get_lines()]
        if not r"$\bar{u}^+=y^+$" in labels and plot_sublayer:
            uplus_max = np.amax([l.get_ydata() for l in ax.get_lines()])
            u_plus_array = np.linspace(0,uplus_max,100)

            ax.cplot(u_plus_array,u_plus_array,label=r"$\bar{u}^+=y^+$",color='r',linestyle='--')

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')

        return fig, ax

    def _get_uuplus_yplus_transforms(self,PhyTime,x_val,y_transform):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x',x_val)[0]
        
        x_transform = lambda y:  y[1:]/delta_v[x_index] if isinstance(y,np.ndarray) else y/delta_v[x_index]
        if y_transform is None:
            y_transform = lambda u: u[1:]/(u_tau[x_index]*u_tau[x_index]) \
                          if isinstance(u,np.ndarray) \
                          else u/(u_tau[x_index]*u_tau[x_index])
        else:
            y_transform = lambda u: u[1:]/(u_tau[x_index]*u_tau[x_index]) \
                          if isinstance(u,np.ndarray) \
                          else u/(u_tau[x_index]*u_tau[x_index])
                          
        return x_transform, y_transform
    
    def plot_Reynolds(self,comp,x_vals,wall_units=False,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            raise ValueError("Reynolds stress component %s not found"%comp) 

        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = check_list_vals(x_vals)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)

        if comp == 'uv':
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        for x in x_vals:
            if wall_units:
                transform_x, transform_y = self._get_uuplus_yplus_transforms(PhyTime,
                                                                            x,
                                                                            transform_y)
            else:
                transform_x = None
            
            
            labels = [self.Domain.create_label(r"$x = %.3g$"%x)] 
            if line_kw is not None:
                if 'label' in line_kw:
                    labels = None

            fig, ax = self.uu_data.plot_line(comp,'y',x,time=PhyTime,labels=labels,
                                             fig=fig,channel_half=True,
                                             transform_ydata=transform_y,
                                             transform_xdata=transform_x,
                                             ax=ax,line_kw=line_kw,**kwargs)

        sign = "-" if comp == 'uv' else ""

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = self.Domain.avgStyle(uu_label)
        y_label = r"$%s %s$"%(sign,avg_label)
        x_label = self.Domain.create_label(r"$y$")
        
        if wall_units:
            y_label = "$%s^+$"%y_label
            x_label = "$y^+$"
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
            if 'label' not in line_kw:
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

    def plot_flatness(self,comp,axis_vals,direction='y',y_mode_wall=False,x_val=0,norm=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        self._check_attr('uuuu_data')
        if comp not in self.uuuu_data.inner_index:
            raise ValueError(f"Component {comp} not found")
        
        axis_vals = check_list_vals(axis_vals)
        
        if direction == 'x':
            if y_mode_wall:
                CoordDF = self._avg_data.Wall_Coords(x_val)
                y = [CoordDF.index_calc('y',x)[0] for x in axis_vals]
            else:
                y = [self.CoordDF.index_calc('y',x)[0] for x in axis_vals]
                
            vals = self.CoordDF['y'][y]
        else:
            x = [self.CoordDF.index_calc('x',x)[0] for x in axis_vals]
            vals = self.CoordDF['x'][x]


        data = self.uuuu_data[PhyTime,[comp]].copy()
        if norm:
            uu = self.uu_data[PhyTime,comp[:2]]
            data = data/(uu*uu)
        
        symbol = 'x' if direction == 'y' else 'wall' if y_mode_wall else 'half-channel'
        unit = get_symbol(symbol)
        labels = [self.Domain.create_label(rf"${unit} = %.3g$"%x) for x in axis_vals] 
        
        if line_kw is not None:
            if 'label' in line_kw:
                labels = None   
                         
        fig, ax = data.plot_line(comp,direction,vals,time=PhyTime,labels=labels,
                                             fig=fig,channel_half=True,
                                             ax=ax,line_kw=line_kw,**kwargs)

        if norm:
            ax.set_ylabel(r"$\overline{%s'^4}/\overline{%s'^2}^2$"%(comp[0],comp[0]))
        else:
            ax.set_ylabel(r"$\overline{%s'^4}$"%(comp[0]))

        x_label = self.Domain.create_label(f"${direction}$")
        ax.set_xlabel(x_label)
                
        return fig, ax

    def plot_skewness(self,comp,axis_vals,direction='y',y_mode_wall=False,x_val=0,norm=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        self._check_attr('uuu_data')
        if comp not in self.uuu_data.inner_index:
            raise ValueError(f"Component {comp} not found")
        
        axis_vals = check_list_vals(axis_vals)
        
        if direction == 'x':
            if y_mode_wall:
                CoordDF = self._avg_data.Wall_Coords(x_val)
                y = [CoordDF.index_calc('y',x)[0] for x in axis_vals]
            else:
                y = [self.CoordDF.index_calc('y',x)[0] for x in axis_vals]
                
            vals = self.CoordDF['y'][y]
        else:
            x = [self.CoordDF.index_calc('x',x)[0] for x in axis_vals]
            vals = self.CoordDF['x'][x]


        data = self.uuu_data[PhyTime,[comp]].copy()
        if norm:
            uu = self.uu_data[PhyTime,comp[:2]]
            data = data/(uu**1.5)
        
        symbol = 'x' if direction == 'y' else 'wall' if y_mode_wall else 'half-channel'
        unit = get_symbol(symbol)
        labels = [self.Domain.create_label(rf"${unit} = %.3g$"%x) for x in axis_vals] 
        
        if line_kw is not None:
            if 'label' in line_kw:
                labels = None   
                         
        fig, ax = data.plot_line(comp,direction,vals,time=PhyTime,labels=labels,
                                             fig=fig,channel_half=True,
                                             ax=ax,line_kw=line_kw,**kwargs)

        if norm:
            ax.set_ylabel(r"$\overline{%s'^3}/\overline{%s'^2}^{3/2}$"%(comp[0],comp[0]))
        else:
            ax.set_ylabel(r"$\overline{%s'^3}$"%(comp[0]))

        x_label = self.Domain.create_label(f"${direction}$")
        ax.set_xlabel(x_label)
                
        return fig, ax
class x3d_avg_xz(_AVG_base,stat_xz_handler):

    def _hdf_extract(self, fn,key=None):
        key = self._get_hdf_key(key)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')

        self.mean_data = fp.FlowStruct1D.from_hdf(fn,key=key+'/mean_data')
        self.uu_data = fp.FlowStruct1D.from_hdf(fn,key=key+'/uu_data')

        h5_obj = fp.hdfHandler(fn,'r',key=key)
        if 'uuu_data' in h5_obj.keys():
            self.uuu_data = fp.FlowStruct1D.from_hdf(fn,key=key+'/uuu_data')
        
        if 'uuuu_data' in h5_obj.keys():
            self.uuuu_data = fp.FlowStruct1D.from_hdf(fn,key=key+'/uuuu_data')                

        return fp.hdfHandler(fn,'r',key=key)

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

    velo_scale_calc = _velo_scale_calc 
    bulk_velo_calc = _velo_scale_calc

    def Cf_calc(self,PhyTime=None):
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
        
        x_transform = lambda y:  y[1:]/delta_v if isinstance(y,np.ndarray) else y/delta_v
        y_transform = lambda u: u[1:]/u_tau if isinstance(u,np.ndarray) else u/u_tau
        
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

            u_plus_array = np.linspace(0,ax.get_ylim()[1],100)
            ax.cplot(u_plus_array,u_plus_array,
                     label=r"$\bar{u}^+=y^+$",
                     color='r',
                     linestyle='--')

        

        ax.set_xscale('log')
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")

        return fig, ax        

    def _get_uuplus_y_plus_transforms(self,PhyTime,y_transform):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        
        x_transform = lambda y:  y[1:]/delta_v if isinstance(y,np.ndarray) else y/delta_v
        if y_transform is None:
            y_transform = lambda u: u[1:]/(u_tau*u_tau) if isinstance(u,np.ndarray) else u/(u_tau*u_tau)
        else:
            y_transform = lambda u: y_transform(u[1:])/(u_tau*u_tau) if isinstance(u,np.ndarray) else u/(u_tau*u_tau)
        return x_transform, y_transform  

    def plot_Reynolds(self,comp,wall_units=False,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            msg = "Reynolds stress component %s not found"%comp
            raise ValueError(msg) 

        PhyTime = self.check_PhyTime(PhyTime)

        if comp == 'uv' and self.Domain.is_channel:
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        if wall_units:
            transform_x, transform_y = self._get_uuplus_y_plus_transforms(PhyTime,
                                                    transform_y)
        else:
            transform_x = None
            
        fig, ax = self.uu_data.plot_line(comp,time=PhyTime,fig=fig,channel_half=True,
                                    transform_ydata=transform_y,
                                    transform_xdata=transform_x,
                                    ax=ax,line_kw=line_kw,**kwargs)

        sign = "-" if comp == 'uv' else ""


        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = self.Domain.avgStyle(uu_label)

        x_label = self.Domain.create_label(r"$y$")
        if wall_units:
            ax.set_xscale('log')
            y_label = "$%s%s^+$"%(sign,avg_label)
            x_label = "$y^+$"
        else:
            y_label = r"$%s %s$"%(sign,avg_label)
            
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
    
    def plot_flatness(self,comp,norm=False,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        self._check_attr('uuuu_data')
        if comp not in self.uuuu_data.inner_index:
            raise ValueError(f"Component {comp} not found")

        PhyTime = self.check_PhyTime(PhyTime)

        data = self.uuuu_data[PhyTime,[comp]].copy()
        if norm:
            uu = self.uu_data[PhyTime,comp[:2]]
            data = data/(uu*uu)
            
        fig, ax = data.plot_line(comp,time=PhyTime,fig=fig,channel_half=True,
                                    ax=ax,line_kw=line_kw,**kwargs)


        x_label = self.Domain.create_label(r"$y$")
        if norm:
            ax.set_ylabel(r"$\overline{%s'^4}/\overline{%s'^2}^2$"%(comp[0],comp[0]))
        else:
            ax.set_ylabel(r"$\overline{%s'^4}$"%(comp[0]))
            
        ax.set_xlabel(x_label)
        
        return fig, ax  
    
    def plot_skewness(self,comp,norm=False,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
            
        self._check_attr('uuu_data')
        if comp not in self.uuu_data.inner_index:
            raise ValueError(f"Component {comp} not found")

        PhyTime = self.check_PhyTime(PhyTime)

        data = self.uuu_data[PhyTime,[comp]].copy()
        if norm:
            uu = self.uu_data[PhyTime,comp[:2]]
            data = data/(uu**1.5)
            
        fig, ax = data.plot_line(comp,time=PhyTime,fig=fig,channel_half=True,
                                    ax=ax,line_kw=line_kw,**kwargs)


        x_label = self.Domain.create_label(r"$y$")
        if norm:
            ax.set_ylabel(r"$\overline{%s'^3}/\overline{%s'^2}^{3/2}$"%(comp[0],comp[0]))
        else:
            ax.set_ylabel(r"$\overline{%s'^3}$"%(comp[0]))
            
        ax.set_xlabel(x_label)
        
        return fig, ax  
    
class x3d_avg_xzt(_AVG_developing,stat_xzt_handler,x3d_avg_xz,CommonTemporalData):

    @classmethod
    def from_phase_average(cls,paths,its=None,*args,**kwargs):
        its_list = cls._get_its_phase(paths,its=its)
        avg_list = []
        for path,its in zip(paths,its_list):
            avg = cls(path,its=its,*args,**kwargs)
            
            avg._test_times_shift(path)
            avg_list.append(avg)
            
        return cls.phase_average(*avg_list)
                
    def _extract_umean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)
        return super()._extract_umean(path,its,None)

    def _extract_uumean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)
        return super()._extract_uumean(path,its,None)
    
    def _extract_avg(self,path,its=None):
        if its is not None: its = check_list_vals(its)
        super()._extract_avg(its,path,None)
    
    def _hdf_extract(self, fn,key=None):
        key = self._get_hdf_key(key)

        self._meta_data = self._module._meta_class.from_hdf(fn,key=key+'/meta_data')

        self.mean_data = fp.FlowStruct1D_time.from_hdf(fn,key=key+'/mean_data')
        self.uu_data = fp.FlowStruct1D_time.from_hdf(fn,key=key+'/uu_data')

        h5_obj = fp.hdfHandler(fn,'r',key=key)
        if 'uuu_data' in h5_obj.keys():
            self.uuu_data = fp.FlowStruct1D_time.from_hdf(fn,key=key+'/uuu_data')
        
        if 'uuuu_data' in h5_obj.keys():
            self.uuuu_data = fp.FlowStruct1D_time.from_hdf(fn,key=key+'/uuuu_data')                

        return fp.hdfHandler(fn,'r',key=key)

    def check_PhyTime(self, PhyTime):
        min_t = np.min(self.times) 
        max_t = np.max(self.times) 
        if PhyTime > max_t or PhyTime < min_t:
            raise ValueError(f"Time given ({PhyTime}) "
                             f"outside range [{min_t}, {max_t}]")
            
        index = np.argmin(np.abs(self.times-PhyTime))
        return self.times[index]
    def _return_index(self,PhyTime):
        if not isinstance(PhyTime,str):
            PhyTime = "{:.9g}".format(PhyTime)

        if PhyTime not in self.get_times():
            raise ValueError("time %s must be in times"% PhyTime)
        for i in range(len(self.get_times())):
            if PhyTime==self.get_times()[i]:
                return i
            
    def _return_xaxis(self):
        return self.times
    
    def wall_unit_calc(self,PhyTime=None):
        """
        returns arrays for the friction velocity and viscous lengthscale


        Returns
        -------
        %(ndarray)s
            friction velocity array
        %(ndarray)s
            viscous lengthscale array
        """
        if PhyTime is None:
            return self._wall_unit_calc(None)
        else:
            return super().wall_unit_calc(PhyTime)
        
    def int_thickness_calc(self,PhyTime=None):
        """
        Calculates the integral thicknesses and shape factor 

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        if PhyTime is None:
            return self._int_thickness_calc(None)
        else:
            return super().int_thickness_calc(PhyTime)

    def Wall_Coords(self,axis_val,PhyTime=None):
        _, delta_v = self.wall_unit_calc(PhyTime)
        index = self._return_index(axis_val)
        return  self.CoordDF / delta_v[index]

    def get_coords_wall_units(self,comp,coords,axis_val,PhyTime=None):
        indices = self.Wall_Coords(axis_val,PhyTime).index_calc(comp,coords)
        return self.CoordDF[comp][indices]            
    
    def plot_shape_factor(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the shape factor from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        _,_, H = self.int_thickness_calc(None)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = update_line_kw(line_kw,label=r"$H$")
        ax.cplot(times,H,**line_kw)

        ax.set_xlabel(r"$%s$"% self.Domain.timeStyle)
        ax.set_ylabel(r"$H$")
        
        return fig, ax    
    
    def plot_disp_thickness(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the displacement thickness from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        delta,_, _ = self.int_thickness_calc(None)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = update_line_kw(line_kw,label=r"$\delta^*$")
        ax.cplot(times,delta,**line_kw)

        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        ax.set_ylabel(r"$\delta^*$")
        
        return fig, ax
    
    def plot_mom_thickness(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the momentum thickness from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        _,theta, _ = self.int_thickness_calc(None)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = update_line_kw(line_kw,label=r"$\theta$")
        ax.cplot(times,theta,**line_kw)

        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        ax.set_ylabel(r"$\theta$")
        
        return fig, ax
    
    def velo_scale_calc(self,PhyTime=None):
        """
        Method to calculate the bulk velocity against the streamwise coordinate
        
        Returns
        -------
        %(ndarray)s
            array containing the bulk velocity
        """
        if PhyTime is None:
            return self._velo_scale_calc(None)
        else:
            return super().velo_scale_calc(PhyTime)
        
    bulk_velo_calc = velo_scale_calc
    def plot_bulk_velocity(self,fig=None,ax=None,line_kw=None,**kwargs):
    
        bulk_velo = self.bulk_velo_calc(None)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times
        line_kw = update_line_kw(line_kw,label=r"$U_{b0}$")

        ax.cplot(times,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        
        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax        
    
    def tau_calc(self,PhyTime=None):
        """
        method to return the wall shear stress array

        Returns
        -------
        %(ndarray)s
            Wall shear stress array 
        """
        if PhyTime is None:
            return self._tau_calc(None)
        else:
            return super().tau_calc(PhyTime)
        
    def plot_skin_friction(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
    
        skin_friction = self._Cf_calc(PhyTime)
        times = self.times

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = update_line_kw(line_kw,label=r"$C_f$")
        ax.cplot(times,skin_friction,**line_kw)

        ax.set_ylabel(r"$C_f$")

        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax
    
    def plot_mean_flow(self,comp,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
        line_kw = update_line_kw(line_kw)

        for time in PhyTimes:

            time_label = self.Domain.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_mean_flow(comp,time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        return fig, ax
    
    def plot_flow_wall_units(self,PhyTimes,fig=None,ax=None,line_kw=None,plot_sublayer=True,**kwargs):
        line_kw = update_line_kw(line_kw)

        for time in PhyTimes:
            
            time_label = self.Domain.timeStyle
            if 'label' not in line_kw:
                line_kw['label'] = r"$%s = %.3g$"%(time_label,time)
            
            sublayer = True if time == PhyTimes[-1] and plot_sublayer else False
            fig, ax = super().plot_flow_wall_units(time,
                                                   fig=fig,
                                                   ax=ax,
                                                   line_kw=line_kw,
                                                   plot_sublayer=sublayer,
                                                   **kwargs)
        
        return fig, ax
    
    def plot_Reynolds(self,comp,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
        line_kw = update_line_kw(line_kw)
        for time in PhyTimes:
            
            time_label = self.Domain.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_Reynolds(comp,PhyTime=time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        return fig, ax
    
    def plot_eddy_visc(self,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
    
        line_kw = update_line_kw(line_kw)
        for time in PhyTimes:
            time_label = self.Domain.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_eddy_visc(time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        return fig, ax
    
    def plot_Reynolds_x(self,comp,y_vals_list,y_mode='half-channel',fig=None,ax=None,line_kw=None,**kwargs):
        
        comp = ''.join(sorted(comp))
        if comp not in self.uu_data.inner_index:
            raise ValueError("Reynolds stress component %s not found"%comp) 


        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = self.Domain.avgStyle(uu_label)

        line_kw = update_line_kw(line_kw)
        if y_vals_list == 'max':
            if 'label' not in line_kw:
                line_kw['label'] = r"$%s_{max}$"%avg_label
            
            fig, ax = self.uu_data.plot_line_time_max(comp,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        else:
            msg = "This method needs to be reimplemented only max can currently be used"
            raise NotImplementedError(msg)

        ax.set_ylabel(r"$%s$"%avg_label)

            
        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax    
    
_avg_xzt_class = x3d_avg_xzt