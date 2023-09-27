import numpy as np
import scipy

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

from scipy.interpolate import interp1d

import x3d_post.post as xp
from flowpy.plotting import (create_fig_ax_with_squeeze,
                             update_line_kw,
                             update_subplots_kw)
from numbers import Number


class x3d_avg_z(xp.x3d_avg_z):
    y_limit = None

    def _int_thickness_calc(self, PhyTime, y_limit=None, U0=None, xindex=None):
        if self.y_limit is None and y_limit is None:
            index = -1
        elif self.y_limit is None:
            index = self.CoordDF.index_calc('y', y_limit)[0]
        else:
            index = self.CoordDF.index_calc('y', self.y_limit)[0]

        if U0 is None:
            U0 = self.mean_data[PhyTime, 'u'][index, np.newaxis]

        if xindex is None:
            U_mean = self.mean_data[PhyTime, 'u'].copy()[:index, slice(None)]
        else:
            U_mean = self.mean_data[PhyTime, 'u'].copy()[:index, xindex]
        y_coords = self.CoordDF['y'][:index]

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = 1 - U_mean/U0

        mom_thickness = integrate_simps(theta_integrand, y_coords, axis=0)
        disp_thickness = integrate_simps(delta_integrand, y_coords, axis=0)
        shape_factor = disp_thickness/mom_thickness

        return disp_thickness, mom_thickness, shape_factor

    def blayer_thickness_calc(self, PhyTime=None, method='classic', thresh=99, dynamic=False):
        PhyTime = self.check_PhyTime(PhyTime)
        if isinstance(method, Number):
            thresh = method
            method = 'classic'

        if method == 'classic':
            return self._delta_classic_calc(PhyTime, threshold=thresh, dynamic=dynamic)
        elif method.lower() == 'griffin':
            return self._delta_griffin(PhyTime, threshold=thresh)
        elif method.lower() == 'diagnostic':
            return self._delta_diagnostic(PhyTime)

    def compute_inviscid_u(self, PhyTime=None):
        p = self.mean_data[PhyTime, 'p']
        v = self.mean_data[PhyTime, 'v']
        u = self.mean_data[PhyTime, 'u']

        p0 = np.amax(p+0.5*u*u)

        return np.sqrt(2*(p0-p) - v*v)

    def _delta_griffin(self, PhyTime, threshold=99):
        u = self.mean_data[PhyTime, 'u']
        U_i = self.compute_inviscid_u(PhyTime)
        u_shift = u[-1, :]-U_i[-1, :]
        U_i += u_shift
        U_i_infty = U_i[-1, :]
        thresh = (1-0.01*threshold)*U_i_infty
        diff = np.abs(u-U_i)
        ids = diff < thresh
        delta99 = np.zeros(u.shape[-1])
        for i, id in enumerate(ids.T):
            for j in range(u.shape[0]):
                if id[j]:
                    y_l = self.CoordDF['y'][j-1]
                    y_u = self.CoordDF['y'][j]
                    ddiff = diff[j, i]-diff[j-1, i]
                    delta99[i] = y_l + (thresh[i]-diff[j-1, i])*(y_u-y_l)/ddiff
                    break
        return delta99

    def _Cf_calc(self, PhyTime, y_ref=None):
        if y_ref is None:
            return super()._Cf_calc(PhyTime)
        else:
            rho_star = 1.0
            re = self.metaDF['re']
            tau_star = self.tau_calc(PhyTime)
            bulk_velo = self.mean_data.slice[y_ref, :]['u']

            skin_friction = (
                2.0/(rho_star*bulk_velo*bulk_velo))*(1/re)*tau_star

            return skin_friction

    def plot_skin_friction(self, y_ref=None, PhyTime=None, fig=None, ax=None, line_kw=None, **kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        skin_friction = self._Cf_calc(PhyTime, y_ref=y_ref)
        x_coords = self.mean_data.CoordDF['x']

        kwargs = update_subplots_kw(kwargs, figsize=[7, 5])
        fig, ax = create_fig_ax_with_squeeze(fig, ax, **kwargs)

        line_kw = update_line_kw(line_kw, label=r"$C_f$")
        ax.cplot(x_coords, skin_friction, **line_kw)
        ax.set_ylabel(r"$C_f$")

        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_blayer_thickness(self, PhyTime=None, method='classic', thresh=99, dynamic=True, fig=None, ax=None, line_kw=None, **kwargs):
        delta = self.blayer_thickness_calc(
            PhyTime, method=method, thresh=thresh, dynamic=dynamic)

        fig, ax = create_fig_ax_with_squeeze(fig=fig, ax=ax, **kwargs)
        x_coords = self.CoordDF['x']

        line_kw = update_line_kw(line_kw, label=r'$\delta$')

        ax.cplot(x_coords, delta, **line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\delta$')

        return fig, ax

    def _delta_classic_calc(self, PhyTime, threshold='99', dynamic=True):
        u_mean = self.mean_data[PhyTime, 'u']
        y_coords = self.CoordDF['y']

        if self.y_limit is None:
            index = -1
        else:
            index = self.CoordDF.index_calc('y', self.y_limit)[0]

        U0 = self.mean_data[PhyTime, 'u'][index, :]

        thresh = float(threshold)*0.01
        delta99 = np.zeros(u_mean.shape[-1])
        for i in range(u_mean.shape[-1]):
            u_99 = thresh*U0[i]
            int = interp1d(u_mean[:index, i], y_coords[:index])

            delta99[i] = int(u_99)
        if dynamic:
            for i in range(u_mean.shape[-1]):
                index = self.CoordDF.index_calc(
                    'y', min(delta99[i]+5, self.y_limit))[0]
                U0 = self.mean_data[PhyTime, 'u'][index, :]
                u_99 = thresh*U0[i]
                int = interp1d(u_mean[:index, i], y_coords[:index])

                delta99[i] = int(u_99)
        return delta99

    def _delta_diagnostic(self, PhyTime):
        u = self.mean_data[PhyTime, 'u']
        uu = np.sqrt(self.uu_data[PhyTime, 'uu'])
        y_coords = self.CoordDF['y']
        delta99_array = np.zeros(u.shape[-1])

        for i in range(u.shape[-1]):
            print(i)
            U_e = np.max(u[:, i])
            H = 1.4
            H_old = 0
            uu_norm = uu[:, i]/(U_e*np.sqrt(H))
            u_norm = u[:, i]/(U_e)

            y_interp = interp1d(uu_norm, y_coords)
            u_interp = interp1d(y_coords, u[:, i])
            delta99 = y_interp(0.02)[()]
            U_e = u_interp(delta99)/0.99

            while abs((H-H_old)/H) > 1e-5:
                print(H, U_e, delta99)
                H_old = H
                _, _, H = self._int_thickness_calc(PhyTime, y_limit=delta99,
                                                   U0=U_e, xindex=i)
                uu_norm = uu[:, i]/(U_e*np.sqrt(H))
                y_interp = interp1d(uu_norm, y_coords)
                delta99 = y_interp(0.02)[()]

                U_e = u_interp(delta99)/0.99
                print(H)

            delta99_array[i] = delta99
        return delta99_array

    def _tau_calc(self, PhyTime):
        u_velo = self.mean_data[PhyTime, 'u']
        ycoords = self.CoordDF['y']

        mu_star = 1.0
        tau_star = mu_star*(u_velo[1, :] - u_velo[0, :]
                            )/(ycoords[1]-ycoords[0])

        return tau_star

    def _velo_scale_calc(self, PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)
        return self.mean_data[PhyTime, 'u'][-1, :].copy()

    U_infty_calc = xp.x3d_avg_z.bulk_velo_calc

    def plot_U_infty(self, *args, **kwargs):
        fig, ax = super().plot_bulk_velocity(*args, **kwargs)
        ax.set_ylabel(r"$U_\infty$")
        return fig, ax

    def _y_plus_calc(self, PhyTime):

        y_coord = self.CoordDF['y']
        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus = y_coord[:, np.newaxis]*delta_v_star
        return y_plus

    def _get_uplus_yplus_transforms(self, PhyTime, x_val):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x', x_val)[0]
        def x_transform(y): return y/delta_v[x_index]
        def y_transform(u): return u/u_tau[x_index]

        return x_transform, y_transform

    def accel_param_calc(self, PhyTime=None, y_ref=None):

        PhyTime = self.check_PhyTime(PhyTime)

        if y_ref is None:
            U_infty = self._velo_scale_calc(PhyTime)
        else:
            U_infty = self.mean_data.slice[y_ref, :]['u']

        U_infty_grad = np.gradient(U_infty, self.mean_data.CoordDF['x'])

        re = self.metaDF['re']

        accel_param = (1/(re*U_infty**2))*U_infty_grad

        return accel_param

    def plot_accel_param(self, y_ref=None, PhyTime=None, desired=False, fig=None, ax=None, line_kw=None, **kwargs):

        accel_param = self.accel_param_calc(PhyTime, y_ref=y_ref)
        x_coords = self.CoordDF['x']

        kwargs = update_subplots_kw(kwargs, figsize=[7, 5])
        fig, ax = create_fig_ax_with_squeeze(fig, ax, **kwargs)

        line_kw = update_line_kw(line_kw, label=r"$K$")

        ax.cplot(x_coords, accel_param, **line_kw)
        if desired:
            re = self.metaDF['re']
            U = self._meta_data.u_infty
            k_des = (1/(re*U**2))*np.gradient(U, x_coords)

            ax.cplot(x_coords, k_des, label=r'$K_{des}$')
        xlabel = self.Domain.create_label(r"$x$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$K$")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(-5, 5))
        return fig, ax


_avg_z_class = x3d_avg_z


class meta_x3d(xp.meta_x3d):
    def _meta_hook(self, path, params):
        self.u_infty = params["tbl_recy"]["u_infty"]

    def save_hdf(self, fn, mode, key=None):
        key = self._get_hdf_key(key)

        h5_obj = super().save_hdf(fn, mode, key)
        h5_obj.create_dataset('u_infty', data=self.u_infty)

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


class line_probes(xp.line_probes):
    pass
