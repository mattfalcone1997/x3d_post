import numpy as np
import pandas as pd
import os
from abc import ABC
import scipy

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

class x3d_avg_z(xp.x3d_avg_z):
    y_limit = None
    def _int_thickness_calc(self,PhyTime):
        if self.y_limit is None:
            index = -1
        else:
            index = self.CoordDF.index_calc('y',self.y_limit)[0]
            
        U0 = self.mean_data[PhyTime,'u'][index,np.newaxis]
        
        U_mean = self.mean_data[PhyTime,'u'].copy()[:index]
        y_coords = self.CoordDF['y'][:index]

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = 1 - U_mean/U0

        mom_thickness = integrate_simps(theta_integrand,y_coords,axis=0)
        disp_thickness = integrate_simps(delta_integrand,y_coords,axis=0)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor
    
    def blayer_thickness_calc(self,PhyTime=None,method=99):
        PhyTime= self.check_PhyTime(PhyTime)

        
        return self._delta99_calc(PhyTime,threshold=method)
        
    
    def plot_blayer_thickness(self,PhyTime=None,method=99,fig=None,ax=None,line_kw=None,**kwargs):
        delta = self.blayer_thickness_calc(PhyTime,method=method)

        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x']

        line_kw = update_line_kw(line_kw,label=r'$\delta$')

        ax.cplot(x_coords,delta,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\delta$')

        return fig, ax

    def _delta99_calc(self,PhyTime,threshold='99'):
        u_mean = self.mean_data[PhyTime,'u']
        y_coords = self.CoordDF['y']
        
        if self.y_limit is None:
            index = -1
        else:
            index = self.CoordDF.index_calc('y',self.y_limit)[0]

        U0 = self.mean_data[PhyTime,'u'][index,:]

        thresh = float(threshold)*0.01
        delta99 = np.zeros(u_mean.shape[-1])
        for i in range(u_mean.shape[-1]):
            u_99 = thresh*U0[i]
            int = interp1d(u_mean[:index,i], y_coords[:index])
            
            delta99[i] = int(u_99)
            
        return delta99
                
        
    def _tau_calc(self,PhyTime):
        u_velo = self.mean_data[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        mu_star = 1.0
        tau_star = mu_star*(u_velo[1,:] - u_velo[0,:])/(ycoords[1]-ycoords[0])        
    
        return tau_star

    def _velo_scale_calc(self, PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)
        return self.mean_data[PhyTime,'u'][-1,:].copy()

    U_infty_calc = xp.x3d_avg_z.bulk_velo_calc
    plot_U_infty = xp.x3d_avg_z.plot_bulk_velocity
    
    def _y_plus_calc(self,PhyTime):

        y_coord = self.CoordDF['y']
        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus = y_coord[:,np.newaxis]*delta_v_star
        return y_plus
    
    def _get_uplus_yplus_transforms(self,PhyTime,x_val):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x',x_val)[0]
        x_transform = lambda y:  y/delta_v[x_index]
        y_transform = lambda u: u/u_tau[x_index]
        
        return x_transform, y_transform
    
    def accel_param_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)

        U_infty = self._velo_scale_calc(PhyTime)
        U_infty_grad = np.gradient(U_infty,self.mean_data.CoordDF['x'])

        re = self.metaDF['re']

        accel_param = (1/(re*U_infty**2))*U_infty_grad
        
        return accel_param

    def plot_accel_param(self,PhyTime=None,desired=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        accel_param = self.accel_param_calc(PhyTime)
        x_coords = self.CoordDF['x']

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = update_line_kw(line_kw,label = r"$K$")
        
        ax.cplot(x_coords,accel_param,**line_kw)
        if desired:
            re = self.metaDF['re']
            U = self._meta_data.u_infty
            k_des = (1/(re*U**2))*np.gradient(U,x_coords)

            ax.cplot(x_coords,k_des,label=r'$K_{des}$')
        xlabel = self.Domain.create_label(r"$x$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$K$")
        
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        return fig,ax

_avg_z_class = x3d_avg_z

class meta_x3d(xp.meta_x3d):
    def _meta_hook(self, path, params):
        self.u_infty = params["tbl_recy"]["u_infty"]

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = super().save_hdf(fn, mode, key)
        h5_obj.create_dataset('u_infty',data=self.u_infty)

    def _hdf_extract(self, fn, key=None):
        key = self._get_hdf_key(key)

        h5_obj = super()._hdf_extract(fn, key)

        self.u_infty = h5_obj['u_infty'][:]

        return h5_obj

_meta_class = meta_x3d


class x3d_inst_z(xp.x3d_inst_z):
    pass

_inst_z_class = x3d_inst_z

class x3d_fluct_z(xp.x3d_fluct_z):
    pass

_fluct_z_class = x3d_fluct_z

class x3d_budget_z(xp.x3d_budget_z):
    pass

class x3d_mom_balance_z(xp.x3d_mom_balance_z):
    pass

class x3d_FIK_z(xp.x3d_FIK_z):
    pass

class x3d_Cf_Renard_z(xp.x3d_Cf_Renard_z):
    pass
class x3d_quadrant_z(xp.x3d_quadrant_z):
    pass

class x3d_spectra_z(xp.x3d_spectra_z):
    pass

class x3d_autocorr_x(xp.x3d_autocorr_x):
    pass