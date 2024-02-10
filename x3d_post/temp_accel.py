from . import post as xp
import types
import numpy as np
from abc import ABC
import flowpy as fp
import os
import scipy
from itertools import product
from numbers import Number
from .post._common import CommonTemporalData

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_line_kw,
                             )
from .utils import get_iterations


class meta_x3d(xp.meta_x3d):
    def _meta_hook(self, path, params):
        if 'profile' in params['temp_accel']:
            if params['temp_accel']['profile'] == 'linear':
                self.metaDF.update({'t_start': params['temp_accel']['t_start'],
                                    't_end': params['temp_accel']['t_end'],
                                    'Re_ratio': params['temp_accel']['Re_ratio']})
            elif params['temp_accel']['profile'] == 'spatial equiv':
                self.metaDF.update({'U_ratio': params['temp_accel']['U_ratio'],
                                    'x0': params['temp_accel']['x0'],
                                    'alpha_accel': params['temp_accel']['alpha_accel']})

        self._tb = np.array(params['temp_accel']['t'])
        self._ub = np.array(params['temp_accel']['U_b'])

        bf_path = os.path.join(path, 'body_force')
        if os.path.isdir(bf_path):
            comp = ['f']
            files = os.listdir(bf_path)
            times = [self.get_time(int(f[-7:])) for f in files]
            data = []
            index = list(product(times, comp))
            for f in files:
                data.append(np.fromfile(os.path.join(bf_path, f), dtype='f8'))

            self.bf = fp.FlowStruct1D_time(
                self.coorddata, np.stack(data, axis=0), index=index)

    def _hdf_extract(self, fn, key=None):
        key = self._get_hdf_key(key)
        hdf_obj = super()._hdf_extract(fn, key)

        self._ub = hdf_obj['ub'][:]
        self._tb = hdf_obj['tb'][:]
        if 'bf' in hdf_obj.keys():
            self.bf = fp.FlowStruct1D_time.from_hdf(fn, key=key+'/bf')

        return hdf_obj

    def save_hdf(self, fn, mode, key=None):
        hdf_obj = super().save_hdf(fn, mode, key)
        hdf_obj.create_dataset('ub', data=self._ub)
        hdf_obj.create_dataset('tb', data=self._tb)
        if hasattr(self, 'bf'):
            self.bf.to_hdf(fn, 'a', key=key+'/bf')

        return hdf_obj


_meta_class = meta_x3d


class temp_accel_base(CommonTemporalData, ABC):
    @classmethod
    def phase_average(cls, *objects, items=None):
        temp_object = super().phase_average(*objects, items=items)

        if 't_start' in temp_object.metaDF:
            temp_object.metaDF['t_start'] += temp_object._time_shift
        if 't_end' in temp_object.metaDF:
            temp_object.metaDF['t_end'] += temp_object._time_shift

        return temp_object

    @property
    def _time_shift(self):
        if 't_start' in self.metaDF:
            return - self.metaDF['t_start']
        else:
            thresh = 1.01
            index = np.argmin(np.abs(self._meta_data._ub-thresh))
            time_ref = self._meta_data._tb[index]
            index = np.argmin(np.abs(self.times-time_ref))
            return -self.times[index]

    def _get_its_shift(cls, path) -> int:
        meta_data = cls._module._meta_class(path)
        if 't_start' in meta_data.metaDF:
            return - meta_data.metaDF['t_start']
        else:
            thresh = 1.01
            index = np.argmin(np.abs(meta_data._ub-thresh))
            time_ref = meta_data._tb[index]
            times = meta_data.get_times(get_iterations(path, statistics=True))
            index = np.argmin(np.abs(times-time_ref))
            return times[index]


class x3d_inst_xzt(xp.x3d_inst_xzt):
    pass


_inst_xzt_class = x3d_inst_xzt


class x3d_fluct_xzt(xp.x3d_fluct_xzt):
    pass


_fluct_xzt_class = x3d_fluct_xzt


class x3d_avg_xzt(xp.x3d_avg_xzt, temp_accel_base):
    @classmethod
    def _type_hook(cls, base_object, attr, vals, time_shifts):
        super()._type_hook(base_object, attr, vals, time_shifts)

        if attr == '_meta_data':
            if hasattr(vals[0], 'bf'):
                bfs = [val.bf.shift_times(shift)
                       for val, shift in zip(vals, time_shifts)]

                times_list = [bf.times for bf in bfs]

                bfs = base_object._handle_time_remove(bfs, times_list)
                vals[0].bf = bfs[0]
                setattr(base_object, attr, vals[0])
            elif hasattr(vals[0], '_tb'):
                tbs = [v._tb for v in vals]
                for t in tbs[1:]:
                    if not np.allclose(tbs[0], t):
                        raise ValueError("Not all times are close")
                new_tb = tbs[0] + time_shifts[0]
                vals[0]._tb = new_tb

            setattr(base_object, attr, vals[0])

    def shift_times(self, time):
        super().shift_times(time)
        self._meta_data._tb += time

    def conv_distance_calc(self, t0=None):

        bulk_velo = self.velo_scale_calc()

        time0 = self.times[0]
        times = [x-time0 for x in self.times]

        start_index = x3d_avg_xzt._return_index(self, t0)
        start_distance = integrate_simps(
            bulk_velo[:start_index], times[:start_index])
        conv_distance = np.zeros_like(bulk_velo)
        for i, _ in enumerate(bulk_velo):
            conv_distance[i] = integrate_simps(
                bulk_velo[:(i+1)], times[:(i+1)])
        return conv_distance - start_distance

    def accel_param_calc(self):
        U_mean = self.mean_data['u']
        U_infty = U_mean[self.NCL[1] // 2]

        times = self.times
        dudt = np.gradient(U_infty, times, edge_order=2)
        REN = self.metaDF['re']

        accel_param = (1/(REN*U_infty**3))*dudt
        return accel_param

    def plot_accel_param(self, fig=None, ax=None, line_kw=None, **kwargs):
        accel_param = self.accel_param_calc()

        xaxis = self._return_xaxis()

        kwargs = update_subplots_kw(kwargs, figsize=[7, 5])
        fig, ax = create_fig_ax_with_squeeze(fig, ax, **kwargs)

        line_kw = update_line_kw(line_kw, label=r"$K$")

        ax.cplot(xaxis, accel_param, **line_kw)

        ax.set_xlabel(r"$t^*$")  # ,fontsize=18)
        ax.set_ylabel(r"$K$")  # ,fontsize=18)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(-5, 5))
        # ax.grid()
        fig.tight_layout()
        return fig, ax

    def _get_data_attr(self):
        data_dict__ = {x: self.__dict__[x] for x in self.__dict__
                       if not isinstance(x, types.MethodType)}
        return data_dict__


_avg_xzt_class = x3d_avg_xzt


class x3d_avg_xzt_conv(x3d_avg_xzt):

    def __init__(self, other_avg: x3d_avg_xzt, t0: Number):

        self.__dict__.update(other_avg._get_data_attr())

        self.CoordDF['x'] = other_avg.conv_distance_calc(t0)
        self._t0 = t0

    def conv_distance_calc(self):
        return super().conv_distance_calc(self._t0)

    def get_times_from_xconv(self, x_conv):
        return self.times[self.CoordDF.index_calc('x', x_conv)]

    def _return_index(self, x_val):
        return self.CoordDF.index_calc('x', x_val)

    def _return_time_index(self, time):
        return super()._return_index(time)

    def _return_xaxis(self):
        return self.CoordDF['x']

    def plot_bulk_velocity(self, *args, **kwargs):
        fig, ax = super().plot_bulk_velocity(*args, **kwargs)
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0], xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_accel_param(self, *args, **kwargs):
        fig, ax = super().plot_accel_param(*args, **kwargs)
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0], xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_skin_friction(self, *args, **kwargs):
        fig, ax = super().plot_skin_friction(*args, **kwargs)
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0], xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_shape_factor(self, *args, **kwargs):
        fig, ax = super().plot_shape_factor(*args, **kwargs)
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)
        ax.set_xlim([xdata[0], xdata[-1]])

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_Reynolds(self, comp, x_vals, *args, **kwargs):
        fig, ax = super().plot_Reynolds(comp, x_vals, *args, **kwargs)
        x_vals = self.CoordDF.get_true_coords('x', x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines, x_vals):
            line.set_xdata(self.CoordDF['x'])
            line.set_label(r"$x_{conv}=%.3g$" % float(x))

        return fig, ax

    def plot_Reynolds_x(self, *args, **kwargs):
        fig, ax = super().plot_Reynolds_x(*args, **kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_bulk_velocity(self, *args, **kwargs):
        fig, ax = super().plot_bulk_velocity(*args, **kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_skin_friction(self, *args, **kwargs):
        fig, ax = super().plot_skin_friction(*args, **kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self, x_vals, *args, **kwargs):
        fig, ax = super().plot_eddy_visc(x_vals, *args, **kwargs)

        x_vals = self.CoordDF.get_true_coords('x', x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines, x_vals):
            line.set_label(r"$x_{conv}=%.3g$" % float(x))

        return fig, ax

    def plot_mean_flow(self, x_vals, *args, **kwargs):
        fig, ax = super().plot_mean_flow(x_vals, *args, **kwargs)

        x_vals = self.CoordDF.get_true_coords('x', x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines, x_vals):
            line.set_label(r"$x_{conv}=%.3g$" % float(x))

        return fig, ax

    def plot_flow_wall_units(self, x_vals, *args, **kwargs):
        fig, ax = super().plot_flow_wall_units(x_vals, *args, **kwargs)

        x_vals = self.CoordDF.get_true_coords('x', x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines, x_vals):
            line.set_label(r"$x_{conv}=%.3g$" % float(x))

        return fig, ax


class x3d_budget_xzt(xp.x3d_budget_xzt, temp_accel_base):
    pass


class x3d_mom_balance_xzt(xp.x3d_mom_balance_xzt, temp_accel_base):
    pass


class x3d_FIK_xzt(xp.x3d_FIK_xzt, temp_accel_base):
    pass


class x3d_spectra_xzt(xp.x3d_spectra_xzt, temp_accel_base):
    pass


class x3d_quadrant_xzt(xp.x3d_quadrant_xzt, temp_accel_base):
    pass


class x3d_Cf_Renard_xzt(xp.x3d_Cf_Renard_xzt, temp_accel_base):
    pass


class x3d_quadrant_xzt(xp.x3d_quadrant_xzt, temp_accel_base):
    pass


class line_probes(xp.line_probes):
    pass


class x3d_correlations_xzt(xp.x3d_correlations_xzt, temp_accel_base):
    pass
