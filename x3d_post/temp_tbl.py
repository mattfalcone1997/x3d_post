import numpy as np
import pandas as pd
import os
from abc import ABC
from numbers import Number
import scipy
import types

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

from scipy.interpolate import interp1d

import x3d_post.post as xp
from x3d_post.post._readers import read_parameters
import flowpy as fp
from flowpy.plotting import (create_fig_ax_with_squeeze,
                             update_line_kw,
                             update_subplots_kw)

class x3d_avg_xzt(xp.x3d_avg_xzt):
    y_limit=None
    def _int_thickness_calc(self,PhyTime):
        U_mean = self.mean_data[PhyTime,'u']
        U0 = U_mean[0]
        y_coords = self.CoordDF['y']

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = U_mean/U0

        mom_thickness = integrate_simps(theta_integrand,y_coords,axis=0)
        disp_thickness = integrate_simps(delta_integrand,y_coords,axis=0)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor
    
    def blayer_thickness_calc(self,PhyTime=None,method=99):
        return self._delta99_calc(PhyTime,threshold=method)
        
    
    def plot_blayer_thickness(self,PhyTime=None,method=99,fig=None,ax=None,line_kw=None,**kwargs):
        delta = self.blayer_thickness_calc(PhyTime,method=method)

        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        line_kw = update_line_kw(line_kw,label=r'$\delta$')

        ax.cplot(self.times,delta,**line_kw)

        xlabel = self.Domain.create_label(r'$t$')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\delta$')

        return fig, ax
    
    def check_PhyTime(self, PhyTime):
        if PhyTime is None:
            return None
        return super().check_PhyTime(PhyTime)
    def _delta99_calc(self,PhyTime,threshold='99'):
        u_mean = self.mean_data[PhyTime,'u']
        y_coords = self.CoordDF['y']
        
        U = self.mean_data[PhyTime,'u'][0] - self.mean_data[PhyTime,'u']

        thresh = float(threshold)*0.01
        delta99 = np.zeros(u_mean.shape[-1])
        for i in range(u_mean.shape[-1]):
            u_99 = thresh*U[-1,i]
            int = interp1d(U[:,i], y_coords)
            
            delta99[i] = int(u_99)
            
        return delta99
    
    def tau_calc(self,PhyTime=None):
        u_velo = self.mean_data[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        mu_star = 1.0
        tau_star = -mu_star*(u_velo[1] - u_velo[0])/(ycoords[1]-ycoords[0])        
        return tau_star

    _tau_calc = tau_calc
    def _velo_scale_calc(self, PhyTime=None):

        return self.mean_data[PhyTime,'u'][0,:].copy()

    wall_velocity_calc = xp.x3d_avg_xzt.bulk_velo_calc
    plot_wall_velocity = xp.x3d_avg_xzt.plot_bulk_velocity

    def _y_plus_calc(self,PhyTime):

        y_coord = self.CoordDF['y']
        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus = y_coord[:,np.newaxis]*delta_v_star
        return y_plus
    
    def _get_uplus_yplus_transforms(self,PhyTime):
        PhyTime = self.check_PhyTime(PhyTime)
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        print(u_tau)
        x_transform = lambda y:  y/delta_v
        y_transform = lambda u: (u[0]-u)/u_tau
        
        return x_transform, y_transform
    
    def accel_param_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)

        U_infty = self._velo_scale_calc(PhyTime)
        dUdt = np.gradient(U_infty,self.times)

        re = self.metaDF['re']

        accel_param = (1/(re*U_infty**3))*dUdt
        
        return accel_param
    
    def plot_accel_param(self,PhyTime=None,desired=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        accel_param = self.accel_param_calc(PhyTime)

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = update_line_kw(line_kw,label = r"$K$")
        
        ax.cplot(self.times,accel_param,**line_kw)
        if desired:
            re = self.metaDF['re']
            U = self._meta_data.wall_velo
            k_des = (1/(re*U**3))*np.gradient(U,self.times)

            ax.cplot(self.times,k_des,label=r'$K_{des}$')

        ax.set_xlabel(r"t")
        ax.set_ylabel(r"$K$")
        
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        return fig,ax
    
    def conv_distance_calc(self,t0=None):
            
        U_w = self.velo_scale_calc()

        time0 = self.times[0]
        times = [x-time0 for x in self.times]

        start_index = x3d_avg_xzt._return_index(self,t0)
        start_distance = integrate_simps(U_w[:start_index],times[:start_index])
        conv_distance = np.zeros_like(U_w)
        for i , _ in enumerate(U_w):
            conv_distance[i] = integrate_simps(U_w[:(i+1)],times[:(i+1)])
        return conv_distance - start_distance
    
    def _get_data_attr(self):
        data_dict__ = {x : self.__dict__[x] for x in self.__dict__ \
                        if not isinstance(x,types.MethodType)}
        return data_dict__

_avg_xzt_class = x3d_avg_xzt
    
class x3d_avg_xzt_conv(x3d_avg_xzt):

    def __init__(self,other_avg: x3d_avg_xzt,t0: Number):
        
        self.__dict__.update(other_avg._get_data_attr())
        
        self.CoordDF['x'] = other_avg.conv_distance_calc(t0)
        self._t0 = t0
        
        
    def conv_distance_calc(self):
        return super().conv_distance_calc(self._t0)
    def get_times_from_xconv(self,x_conv):
        return self.times[self.CoordDF.index_calc('x',x_conv)]
    def _return_index(self,x_val):
        return self.CoordDF.index_calc('x',x_val)

    def _return_time_index(self,time):
        return super()._return_index(time)
    def _return_xaxis(self):
        return self.CoordDF['x']

    def plot_bulk_velocity(self,*args,**kwargs):
        fig, ax = super().plot_bulk_velocity(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0],xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_accel_param(self,*args,**kwargs):
        fig, ax = super().plot_accel_param(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0],xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_skin_friction(self,*args,**kwargs):
        fig, ax = super().plot_skin_friction(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0],xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax    

    def plot_shape_factor(self,*args,**kwargs):
        fig, ax = super().plot_shape_factor(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0],xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_Reynolds(self,comp,x_vals,*args,**kwargs):
        fig, ax = super().plot_Reynolds(comp,x_vals,*args,**kwargs)
        x_vals = self.CoordDF.get_true_coords('x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_xdata(self.CoordDF['x'])
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        
        return fig, ax

    def plot_Reynolds_x(self,*args,**kwargs):
        fig, ax = super().plot_Reynolds_x(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax  

    def plot_bulk_velocity(self,*args,**kwargs):
        fig, ax = super().plot_bulk_velocity(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_skin_friction(self,*args,**kwargs):
        fig, ax = super().plot_skin_friction(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,x_vals, *args, **kwargs):
        fig, ax =  super().plot_eddy_visc(x_vals,*args, **kwargs)

        x_vals = self.CoordDF.get_true_coords('x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))

        return fig, ax

    def plot_mean_flow(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_mean_flow(x_vals,*args,**kwargs)

        x_vals = self.CoordDF.get_true_coords('x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
            
        return fig, ax

    def plot_flow_wall_units(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_flow_wall_units(x_vals,*args,**kwargs)

        x_vals = self.CoordDF.get_true_coords('x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))

        return fig, ax
    
class meta_x3d(xp.meta_x3d):
    def _meta_hook(self, params):
        if "tbltemp_accel" in params:
            self.wall_velo = params["tbltemp_accel"]["U_w"]

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = super().save_hdf(fn, mode, key)
        if hasattr(self,'wall_velo'):
            h5_obj.create_dataset('U_w',data=self.wall_velo)

    def _hdf_extract(self, fn, key=None):
        key = self._get_hdf_key(key)

        h5_obj = super()._hdf_extract(fn, key)
        if 'U_w' in h5_obj.keys():
            self.wall_velo = h5_obj['U_w'][:]

        return h5_obj

_meta_class = meta_x3d    


class x3d_inst_xzt(xp.x3d_inst_xzt):
    pass

_inst_xzt_class = x3d_inst_xzt

# class x3d_fluct_xzt(xp.x3d_fluct_xzt):
#     pass

# _fluct_xzt_class = x3d_fluct_xzt

class x3d_budget_xzt(xp.x3d_budget_xzt):
    pass

class x3d_mom_balance_xzt(xp.x3d_mom_balance_xzt):
    pass

class x3d_FIK_xzt(xp.x3d_FIK_xzt):
    pass

class x3d_Cf_Renard_xzt(xp.x3d_Cf_Renard_xzt):
    pass

class x3d_quadrant_xzt(xp.x3d_quadrant_xzt):
    pass

class x3d_spectra_xzt(xp.x3d_spectra_xzt):
    pass

class x3d_autocorr_x(xp.x3d_autocorr_x):
    pass