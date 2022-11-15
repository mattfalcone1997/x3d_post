from . import post as xp
import types
import numpy as np
from abc import ABC
import scipy

from .post._common import CommonTemporalData

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_line_kw,
                             )

class meta_x3d(xp.meta_x3d):
    def _meta_hook(self, params):
        if params['temp_accel']['profile'] == 'linear':
            self.metaDF.update({'t_start' : params['temp_accel']['t_start'],
                                't_end' : params['temp_accel']['t_end'],
                                'Re_ratio' : params['temp_accel']['Re_ratio']})
        elif params['temp_accel']['profile'] == 'spatial equiv':
            self.metaDF.update({'U_ratio' : params['temp_accel']['U_ratio'],
                                'x0' : params['temp_accel']['x0'],
                                'alpha_accel' : params['temp_accel']['alpha_accel']})
        
_meta_class  = meta_x3d

class temp_accel_base(CommonTemporalData,ABC):
    @classmethod
    def phase_average(cls, *objects, items=None):
        temp_object = super().phase_average(*objects, items=items)
        
        temp_object.metaDF['t_start'] += temp_object._time_shift
        temp_object.metaDF['t_end'] += temp_object._time_shift
        
        return temp_object    
    
    @property
    def _time_shift(self):
        return -self.metaDF['t_start']
    
    def _get_its_shift(cls,path) -> int:
        meta_data = cls._module._meta_class(path)
        
        return meta_data.get_it(meta_data.metaDF['t_start'])
    
class x3d_inst_xzt(xp.x3d_inst_xzt):
    pass

class x3d_avg_xzt(xp.x3d_avg_xzt,temp_accel_base):
    
    def conv_distance_calc(self):
            
        bulk_velo = self.bulk_velo_calc()

        time0 = self.times[0]
        times = [x-time0 for x in self.times]

        accel_start = self.metaDF['temp_start_end'][0]
        start_distance = bulk_velo[0]*(accel_start - time0)
        conv_distance = np.zeros_like(bulk_velo)
        for i , _ in enumerate(bulk_velo):
            conv_distance[i] = integrate_simps(bulk_velo[:(i+1)],times[:(i+1)])
        return conv_distance - start_distance
    
    def accel_param_calc(self):
        U_mean = self.mean_data['u']
        U_infty = U_mean[self.NCL[1] // 2 ]

        times = self.times
        dudt = np.gradient(U_infty,times,edge_order=2)
        REN = self.metaDF['REN']

        accel_param = (1/(REN*U_infty**3))*dudt
        return accel_param
    
    def plot_accel_param(self,fig=None,ax=None,line_kw=None,**kwargs):
        accel_param = self.accel_param_calc()

        xaxis = self._return_xaxis()

        kwargs = update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw = update_line_kw(line_kw,label = r"$K$")


        ax.cplot(xaxis,accel_param,**line_kw)

        ax.set_xlabel(r"$t^*$")# ,fontsize=18)
        ax.set_ylabel(r"$K$")# ,fontsize=18)

        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        #ax.grid()
        fig.tight_layout()
        return fig,ax
    
_avg_xzt_class = x3d_avg_xzt



class x3d_budget_xzt(xp.x3d_budget_xzt,temp_accel_base):
    pass