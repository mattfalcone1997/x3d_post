import numpy as np
from ._data_handlers import (stathandler_base,
                             stat_z_handler, 
                             stat_xz_handler,
                             stat_xzt_handler)

from ._common import (CommonData,
                      classproperty,
                      CommonTemporalData)
from ._average import x3d_avg_z, x3d_avg_xz, x3d_avg_xzt
from ..utils import get_iterations
import flowpy as fp
from flowpy.utils import check_list_vals
from flowpy.plotting import (update_subplots_kw, 
                            create_fig_ax_without_squeeze,
                            create_fig_ax_with_squeeze,
                            update_line_kw,
                            flip_leg_col)

from flowpy.gradient import Scalar_laplacian
from abc import ABC, abstractmethod, abstractproperty
import matplotlib as mpl
from itertools import chain

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    from scipy.integrate import simpson as simps

else:
    from scipy.integrate import cumtrapz,simps
class budgetBase(CommonData,ABC):   

    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._budget_init(*args,**kwargs)

    def _budget_init(self,comp,it,path,it0=None):

        self.avg_data = self._get_avg_data(it,path,it0)
        
        self._get_stat_data(it,path,it0)

        self.budget_data = self._budget_extract(comp)

        self._del_stat_data()

    def _hdf_extract(self,fn,key=None):
        key =self._get_hdf_key(key)

        self.avg_data = self._get_avg_data.from_hdf(fn,key=key+'/avg_data')
        self.budget_data = self._flowstruct_class.from_hdf(fn,key=key+'/budget_data')

    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)

    def save_hdf(self,fn,mode,key=None):
        key =self._get_hdf_key(key)

        self.avg_data.save_hdf(fn,mode,key=key+'/avg_data')
        self.budget_data.to_hdf(fn,'a',key=key+'/budget_data')

    @abstractmethod
    def _get_stat_data(self):
        pass

    @property
    def Domain(self):
        return self.avg_data.Domain

    @property
    def _coorddata(self):
        return self.avg_data._coorddata

    @abstractmethod
    def _del_stat_data(self):
        pass

    @abstractmethod
    def _budget_extract(self,comp,it,path,it0):
        pass

    @property
    def _meta_data(self):
        return self.avg_data._meta_data

    @property
    def shape(self):
        return self.avg_data.shape
    
    def _check_terms(self,comp):
        
        budget_terms = sorted(self.budget_data.inner_index)

        if comp is None:
            comp_list = budget_terms
        elif isinstance(comp,(tuple,list)):
            comp_list = comp
        elif isinstance(comp,str):
            comp_list = [comp]
        else:
            raise TypeError("incorrect time")
        
        for comp in comp_list:
            if not comp in budget_terms:
                raise KeyError(f"Invalid budget term ({comp}) provided")

        return comp_list

    @property
    def Balance(self):
        times = list(self.avg_data.times)
        total_balance = []
        for time in times:
            balance = []
            for term in self.budget_data.inner_index:
                balance.append(self.budget_data[time,term])
            total_balance.append(np.array(balance).sum(axis=0))

        index = [times,['balance']*len(times)]
        return self._flowstruct_class(self._coorddata,
                                        np.array(total_balance),
                                        index=index)
            
    def __str__(self):
        return self.budget_data.__str__()

    def _create_budget_axes(self,x_list,fig=None,ax=None,**kwargs):

        ax_size = len(x_list)
        ax_size=(int(np.ceil(ax_size/2)),2) if ax_size >2 else (ax_size,1)

        lower_extent= 0.2
        gridspec_kw = {'bottom': lower_extent}
        figsize= [7*ax_size[1],5*ax_size[0]+1]

        kwargs = update_subplots_kw(kwargs,gridspec_kw=gridspec_kw,figsize=figsize)
        fig, ax, single_input = create_fig_ax_without_squeeze(*ax_size,fig=fig,ax=ax,**kwargs)
        return fig, ax.flatten(), single_input

    @staticmethod
    def title_with_math(string):
        math_split = string.split('$')
        for i, split in enumerate(math_split):
            if i%2 == 0:
                if math_split[i] != math_split[i].upper():
                    math_split[i] = math_split[i].title()
        return "$".join(math_split)

class ReynoldsBudget_base(stathandler_base,ABC):
    def _budget_extract(self,comp):

        production = self._production_extract(comp)
        advection = self._advection_extract(comp)
        turb_transport = self._turb_transport(comp)
        pressure_diffusion = self._pressure_diffusion(comp)
        pressure_strain = self._pressure_strain(comp)
        viscous_diff = self._viscous_diff(comp)
        dissipation = self._dissipation_extract(comp)
    
        array_concat = [production,advection,turb_transport,pressure_diffusion,\
                        pressure_strain,viscous_diff,dissipation]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['production','advection','turbulent transport','pressure diffusion',\
                     'pressure strain','viscous diffusion','dissipation']  
        phystring_index = [None]*7

        budget_data = self._flowstruct_class(self._coorddata,
                                        budget_array,
                                        index =[phystring_index,budget_index])
        
        return budget_data

    @abstractproperty
    def _get_avg_data(self,it,path,it0):
        pass
    
    @abstractmethod
    def _production_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _advection_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _turb_transport(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _pressure_diffusion(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _pressure_strain(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _viscous_diff(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _dissipation_extract(self,*args,**kwargs):
        raise NotImplementedError        

    def _get_stat_data(self,it, path, it0):
        
        self.mean_data = self.avg_data.mean_data
        self.uu_data = self.avg_data.uu_data

        self.uuu_data = self._extract_uuumean(path,it, it0)
        self.dudx_data = self._extract_dudxmean(path,it, it0)
        self.pu_data = self._extract_pumean( path,it, it0)
        self.pdudx_data = self._extract_pdudxmean(path, it, it0)
        self.dudx2_data = self._extract_dudx2mean(path, it, it0)

    def _del_stat_data(self):
        del self.mean_data
        del self.uu_data
        del self.uuu_data
        del self.dudx_data
        del self.pu_data
        del self.pdudx_data
        del self.dudx2_data

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

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uuu','uuv','uuw','uvv','uvw','uww',
                 'vvv','vvw','vww','www']

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

    def _extract_dudxmean(self,path,it,it0):
        names = ['dudxmean','dudymean','dudzmean',
                 'dvdxmean','dvdymean','dvdzmean',
                 'dwdxmean','dwdymean','dwdzmean']

        l = self._get_data(path,'dudx_mean',names,it,9)
        if it0 is not None:
            l0 = self._get_data(path,'dudx_mean',names,it0,9)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['dudx','dudy','dudz',
                 'dvdx','dvdy','dvdz',
                 'dwdx','dwdy','dwdz']

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        dudx_mean =  self._flowstruct_class(coorddata,
                                            l,
                                            index=index)  

        return dudx_mean

    def _extract_pumean(self,path,it,it0):
        names = ['pumean','pvmean','pwmean']

        l = self._get_data(path,'pu_mean',names,it,3)
        if it0 is not None:
            l0 = self._get_data(path,'pu_mean',names,it0,3)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['pu','pv','pw']

        self._check_attr('mean_data')
        i=0
        for time in self.mean_data.times:
            p = self.mean_data[time,'p']
            for comp in comps:
                comp = comp[1]
                u = self.mean_data[time,comp]
                l[i] = l[i] - p*u
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        pu_data =  self._flowstruct_class(coorddata,
                                          l,
                                          index=index)
        return pu_data

    def _extract_pdudxmean(self,path,it,it0):
        names = ['pdudxmean','pdvdymean','pdwdzmean']

        l = self._get_data(path,'pdudx_mean',names,it,3)
        if it0 is not None:
            l0 = self._get_data(path,'pdudx_mean',names,it0,3)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['pdudx','pdvdy','pdwdz']

        self._check_attr('mean_data')
        self._check_attr('dudx_data')
        i = 0
        for time in self.mean_data.times:
            p = self.mean_data[time,'p']
            for comp in comps:
                comp = comp[1:]
                u = self.dudx_data[time,comp]
                l[i] = l[i] - p*u
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        pdudx_data =  self._flowstruct_class(coorddata,
                                             l,
                                             index=index)
        return pdudx_data

    def _extract_dudx2mean(self,path,it,it0):
        names = ['dudxdudxmean','dudxdvdxmean', 'dudxdwdxmean', 
                'dvdxdvdxmean', 'dvdxdwdxmean', 'dwdxdwdxmean',
                'dudydudymean', 'dudydvdymean', 'dudydwdymean', 
                'dvdydvdymean', 'dvdydwdymean', 'dwdydwdymean']
        
        l = self._get_data(path,'dudxdudx_mean',names,it,12)
        
        if it0 is not None:
            l0 = self._get_data(path,'dudxdudx_mean',names,it0,12)
            
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['dudxdudx','dudxdvdx', 'dudxdwdx', 
                'dvdxdvdx', 'dvdxdwdx', 'dwdxdwdx',
                'dudydudy', 'dudydvdy', 'dudydwdy', 
                'dvdydvdy', 'dvdydwdy', 'dwdydwdy']

        self._check_attr('dudx_data')
        i = 0
        for time in self.mean_data.times:
            for comp in comps:
                comp1 = comp[:4]
                comp2 = comp[4:]

                u1 = self.dudx_data[time,comp1]
                u2 = self.dudx_data[time,comp2]

                l[i] = l[i] - u1*u2
                i += 1
                
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        dudx2_data =  self._flowstruct_class(coorddata,
                                             l,
                                             index=index)
        return dudx2_data  
_avg_z_class = x3d_avg_z
class x3d_budget_z(ReynoldsBudget_base,budgetBase,stat_z_handler):

    @classproperty
    def _get_avg_data(self):
        return self._module._avg_z_class
    
    def _advection_extract(self,comp):

        uu = self.uu_data[comp]
        U_mean = self.mean_data['u']
        V_mean = self.mean_data['v']

        uu_dx = self.Domain.Grad_calc(self.CoordDF,uu,'x')
        uu_dy = self.Domain.Grad_calc(self.CoordDF,uu,'y')

        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection

    def _turb_transport(self,comp):
        uu_comp1 = comp+'u'
        uu_comp2 = comp+'v'

        uu_comp1 = ''.join(sorted(uu_comp1))
        uu_comp2 = ''.join(sorted(uu_comp2))

        u1u2u = self.uuu_data[uu_comp1]
        u1u2v = self.uuu_data[uu_comp2]

        u1u2u_dx = self.Domain.Grad_calc(self.CoordDF,u1u2u,'x')
        u1u2v_dy = self.Domain.Grad_calc(self.CoordDF,u1u2v,'y')

        turb_transport = -(u1u2u_dx + u1u2v_dy)
        return turb_transport

    def _pressure_strain(self,comp):
        u1u2 = 'pd' + comp[0] + 'd'+ chr(ord(comp[1])-ord('u')+ord('x'))
        u2u1 = 'pd' + comp[1] + 'd'+ chr(ord(comp[0])-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.pdudx_data[u1u2]
        pdu2dx1 = self.pdudx_data[u2u1]

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain

    def _pressure_diffusion(self,comp):
        comp1, comp2 = sorted(comp)

        if comp1 == 'w' and comp2 == 'w':
            return np.zeros(self.shape)

        diff1 = chr(ord(comp1)-ord('u')+ord('x'))
        diff2 = chr(ord(comp2)-ord('u')+ord('x'))

        pu1 = self.pu_data['p'+comp1]
        pu2 = self.pu_data['p'+comp2]

        rho_star = 1.0
        pu1_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu1,diff1)
        pu2_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu2,diff2)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff

    def _viscous_diff(self,comp):
        u1u2 = self.uu_data[comp]

        re = self.metaDF['re']
        viscous_diff = (1/re)*Scalar_laplacian(self.CoordDF,u1u2)
        return viscous_diff

    def _production_extract(self,comp):
        comp1, comp2 = sorted(comp)

        U1U_comp = ''.join(sorted(comp1 + 'u'))
        U2U_comp = ''.join(sorted(comp2 + 'u'))
        U1V_comp = ''.join(sorted(comp1 + 'v'))
        U2V_comp = ''.join(sorted(comp2 + 'v'))
        
        u1u = self.uu_data[U1U_comp]
        u2u = self.uu_data[U2U_comp]
        u1v = self.uu_data[U1V_comp]
        u2v = self.uu_data[U2V_comp]

        U1x_comp = 'd' + comp1 + 'd' +  'x'
        U2x_comp = 'd' + comp2 + 'd' + 'x'
        U1y_comp = 'd' + comp1 + 'd' + 'y'
        U2y_comp = 'd' + comp2 + 'd' + 'y'
        
        du1dx = self.dudx_data[U1x_comp]
        du2dx = self.dudx_data[U2x_comp]
        du1dy = self.dudx_data[U1y_comp]
        du2dy = self.dudx_data[U2y_comp]

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production

    def _dissipation_extract(self,comp):
        comp1, comp2 = comp

        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.dudx2_data[dU1dxdU2dx_comp]
        du1dydu2dy = self.dudx2_data[dU1dydU2dy_comp]

        re = self.avg_data.metaDF['re']
        dissipation = -(2/re)*(du1dxdu2dx + du1dydu2dy)
        return dissipation

    def _wallunit_generator(self,x_index,PhyTime,wall_units):

        if wall_units:
            u_tau, delta_v = self.avg_data.wall_unit_calc(PhyTime)
            budget_scale = u_tau**3/delta_v

        def _x_Transform(data):
            return (data.copy())/delta_v[x_index]


        def _y_Transform(data):
            return data/budget_scale[x_index]

        if wall_units:
            return _x_Transform, _y_Transform
        else:
            return None, None

    def plot_budget(self, x_list,PhyTime=None,budget_terms=None, wall_units=True,fig=None, ax =None,line_kw=None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = check_list_vals(x_list)
        x_list = self.CoordDF.get_true_coords('x',x_list)

        budget_terms = self._check_terms(budget_terms)

        fig, ax, single_input = self._create_budget_axes(x_list,fig=fig,ax=ax,**kwargs)
        line_kw= update_line_kw(line_kw)


        for i,x_loc in enumerate(x_list):
            x = self.CoordDF.index_calc('x',x_loc)[0]
            y_plus, budget_scale = self._wallunit_generator(x,PhyTime,wall_units)
            for comp in budget_terms:
                
                line_kw['label'] = self.title_with_math(comp)
                fig, ax[i] = self.budget_data.plot_line(comp,'y',x_loc,time=PhyTime,channel_half=True,
                                                    transform_xdata=y_plus,
                                                    transform_ydata=budget_scale,
                                                    fig=fig,ax=ax[i],line_kw=line_kw)
            
            title = self.Domain.create_label(r"$x = %.1f$"%x_loc)
            ax[i].set_title(title,loc='right')

            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")

            if wall_units:
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r"$y^+$")

            else:
                x_label = self.Domain.create_label(r"$y$")
                ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
            
        return fig, ax[0] if single_input else ax

    def plot_integral_budget(self, budget_terms, PhyTime=None, fig=None, ax=None, line_kw=None, **kwargs):
        budget_terms = self._check_terms(budget_terms)
    
        kwargs = update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= update_line_kw(line_kw)

        x_coords = self.CoordDF['x']

        for comp in budget_terms:
            budget_term = self.budget_data[PhyTime,comp]
            int_budget = 0.5*self.Domain.Integrate_tot(self.Coord_ND_DF,budget_term)
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(x_coords,int_budget,label=label,**line_kw)

        ax.set_xlabel(r"$x/\delta$")        

        return fig, ax     
        
_avg_xz_class = x3d_avg_xz
class x3d_budget_xz(ReynoldsBudget_base,budgetBase,stat_xz_handler):
    _flowstruct_class = fp.FlowStruct1D

    def _extract_uuumean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)        
        return super()._extract_uuumean(path,its,None)

    def _extract_dudxmean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)        
        return super()._extract_dudxmean(path,its,None)

    def _extract_pumean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)        
        return super()._extract_pumean(path,its,None)

    def _extract_pdudxmean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)        
        return super()._extract_pdudxmean(path,its,None)

    def _extract_dudx2mean(self,path,its,it0):
        if its is None:
            its = get_iterations(path,statistics=True)        
        return super()._extract_dudx2mean(path,its,None)
    
    @classproperty
    def _get_avg_data(self):
        return self._module._avg_xz_class

    def _advection_extract(self,comp):
    
        uu = self.avg_data.uu_data[comp]
        V_mean = self.avg_data.mean_data['v']

        uu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -V_mean*uu_dy
        return advection        

    def _turb_transport(self,comp):
        uu_comp = comp+'v'
        uu_comp = ''.join(sorted(uu_comp))        

        u1u2v = self.uuu_data[uu_comp]
        return -self.Domain.Grad_calc(self.CoordDF,u1u2v,'y')
        
    def _pressure_strain(self,comp):
        u1u2 = 'pd' + comp[0] + 'd'+ chr(ord(comp[1])-ord('u')+ord('x'))
        u2u1 = 'pd' + comp[1] + 'd'+ chr(ord(comp[0])-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.pdudx_data[u1u2]
        pdu2dx1 = self.pdudx_data[u2u1]

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain        

    def _pressure_diffusion(self,comp):
        comp1, comp2 = sorted(comp)

        if comp1 == 'w' and comp2 == 'w':
            return np.zeros(self.shape)

        diff1 = chr(ord(comp1)-ord('u')+ord('x'))
        diff2 = chr(ord(comp2)-ord('u')+ord('x'))

        pu1 = self.pu_data['p'+comp1]
        pu2 = self.pu_data['p'+comp2]

        rho_star = 1.0

        if diff1 == 'y':
            pu1_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu1,diff1)
        else:
            pu1_grad = np.zeros_like(pu1)

        if diff2 == 'y':
            pu2_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu2,diff2)
        else:
            pu2_grad = np.zeros_like(pu1)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff        

    def _viscous_diff(self,comp):
        u1u2 = self.uu_data[comp]

        re = self.metaDF['re']
        duu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,u1u2,'y')
        duu_dy2= self.Domain.Grad_calc(self.avg_data.CoordDF,duu_dy,'y')
        return (1/re)*duu_dy2

    def _production_extract(self,comp):
        comp1, comp2 = sorted(comp)

        U1U_comp = ''.join(sorted(comp1 + 'u'))
        U2U_comp = ''.join(sorted(comp2 + 'u'))
        U1V_comp = ''.join(sorted(comp1 + 'v'))
        U2V_comp = ''.join(sorted(comp2 + 'v'))
        
        u1u = self.uu_data[U1U_comp]
        u2u = self.uu_data[U2U_comp]
        u1v = self.uu_data[U1V_comp]
        u2v = self.uu_data[U2V_comp]

        U1x_comp = 'd' + comp1 + 'd' +  'x'
        U2x_comp = 'd' + comp2 + 'd' + 'x'
        U1y_comp = 'd' + comp1 + 'd' + 'y'
        U2y_comp = 'd' + comp2 + 'd' + 'y'
        
        du1dx = self.dudx_data[U1x_comp]
        du2dx = self.dudx_data[U2x_comp]
        du1dy = self.dudx_data[U1y_comp]
        du2dy = self.dudx_data[U2y_comp]

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production            

    def _dissipation_extract(self,comp):
        comp1, comp2 = comp

        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.dudx2_data[dU1dxdU2dx_comp]
        du1dydu2dy = self.dudx2_data[dU1dydU2dy_comp]

        re = self.avg_data.metaDF['re']
        dissipation = -(2/re)*(du1dxdu2dx + du1dydu2dy)
        return dissipation        

    def _wallunit_generator(self,PhyTime,wall_units):
    
        if wall_units:
            u_tau, delta_v = self.avg_data.wall_unit_calc(PhyTime)
            budget_scale = u_tau**3/delta_v

        def _x_Transform(data):
            return data.copy()/delta_v

        def _y_Transform(data):
            return data/budget_scale

        if wall_units:
            return _x_Transform, _y_Transform
        else:
            return None, None        

    def plot_budget(self, PhyTime=None,budget_terms=None, wall_units=True,fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        budget_terms = self._check_terms(budget_terms)
        line_kw= update_line_kw(line_kw)

        y_plus, budget_scale = self._wallunit_generator(PhyTime,wall_units)

        for comp in budget_terms:
                
            line_kw['label'] = self.title_with_math(comp)
            fig, ax = self.budget_data.plot_line(comp,time=PhyTime,
                                                transform_xdata=y_plus,
                                                transform_ydata=budget_scale,
                                                channel_half=True,
                                                fig=fig,ax=ax,line_kw=line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        if wall_units:
            ax.set_xscale('log')
            ax.set_xlabel(r"$y^+$")

        else:
            x_label = self.Domain.create_label(r"$y$")
            ax.set_xlabel(x_label)

        return fig, ax            
_avg_xzt_class = x3d_avg_xzt

class budget_xzt_base(budgetBase,CommonTemporalData):
    @classmethod
    def from_phase_average(cls,comp,paths,its=None,*args,**kwargs):
        its_list = cls._get_its_phase(paths,its=its)
        avg_list = []
        for path,its in zip(paths,its_list):
            avg = cls(comp,path,its=its,*args,**kwargs)
            
            avg._test_times_shift(path)
            avg_list.append(avg)
            
        return cls.phase_average(*avg_list)
    
    @property
    def times(self):
        return self.avg_data.times
    
    @classproperty
    def _get_avg_data(self):
        return self._module._avg_xzt_class
    
    def _budget_init(self, comp, path, its=None):
            
        self.avg_data = self._get_avg_data(path,its=its)
        
        self._get_stat_data(its,path)

        self.budget_data = self._budget_extract(comp)

        self._del_stat_data()
    
class x3d_budget_xzt(x3d_budget_xz,stat_xzt_handler,budget_xzt_base):
    _flowstruct_class = fp.FlowStruct1D_time
    def _budget_extract(self,comp):
        
        transient = self._transient_extract(comp).T
        production = self._production_extract(comp).T
        advection = self._advection_extract(comp).T
        turb_transport = self._turb_transport(comp).T
        pressure_diffusion = self._pressure_diffusion(comp).T
        pressure_strain = self._pressure_strain(comp).T
        viscous_diff = self._viscous_diff(comp).T
        dissipation = self._dissipation_extract(comp).T
    
        array_concat = [transient,production,advection,turb_transport,pressure_diffusion,\
                        pressure_strain,viscous_diff,dissipation]

        budget_array = np.concatenate(array_concat,axis=0)
        
        comps = ['transient','production','advection','turbulent transport','pressure diffusion',\
                     'pressure strain','viscous diffusion','dissipation']  
        
        times = self.avg_data.times
        comps = [[x]*len(times) for x in comps]        
        index = [list(times)*len(comps),list(chain(*comps))]
        
        budget_data = self._flowstruct_class(self._coorddata,
                                        budget_array,
                                        index =index)
        
        return budget_data
            
    def _transient_extract(self,comp):
        uu = self.uu_data[comp]
        times = self.avg_data.times
        index = [times,[comp]*len(times)]
        return-1.*np.gradient(uu,times,axis=-1)
    
    def _get_stat_data(self, its, path):
        return super()._get_stat_data(its, path, None)
    

    def plot_budget(self, time_list,budget_terms=None,wall_units=True, fig=None, ax =None,line_kw=None,**kwargs):
        
        budget_terms = self._check_terms(budget_terms)
        line_kw= update_line_kw(line_kw)
        fig, ax, single_input = self._create_budget_axes(time_list,fig,ax,**kwargs)

        for i,time in enumerate(time_list):
            fig, ax[i] = super().plot_budget(time,
                                            budget_terms=budget_terms,
                                            wall_units=wall_units,
                                            fig=fig,
                                            ax=ax[i],
                                            line_kw=line_kw)
            
            time_label = self.Domain.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')
            
        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax[0] if single_input else ax

    def plot_integral_budget(self, budget_terms, fig=None, ax=None, line_kw=None, **kwargs):
        budget_terms = self._check_terms(budget_terms)
    
        kwargs = update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= update_line_kw(line_kw)

        times = self.avg_data.times

        for comp in budget_terms:
            budget_term = self.budget_data[comp]
            int_budget = 0.5*self.Domain.Integrate_tot(self.Coord_ND_DF,budget_term)
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(times,int_budget,label=label,**line_kw)

        time_label = self.Domain.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)
        
        return fig, ax    
    
    
class momentum_balance_base(budgetBase):
    def __init__(self,comp,avg_data):
        
        self._get_stat_data(avg_data)

        self.budget_data = self._budget_extract(comp)

        self._del_stat_data()
        
    def _get_stat_data(self,avg_data):
        
        self.avg_data = avg_data
        self.mean_data = self.avg_data.mean_data
        self.uu_data = self.avg_data.uu_data


    def _del_stat_data(self):
        del self.mean_data
        del self.uu_data
                
    def _budget_extract(self,comp):
            
        advection = self._advection_extract(comp)
        pressure_grad = self._pressure_grad(comp)
        viscous = self._viscous_extract(comp)
        Reynolds_stress = self._turb_transport(comp)


        array_concat = [advection,pressure_grad,viscous,Reynolds_stress]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['advection','pressure gradient','viscous stresses','reynolds stresses']  
        phystring_index = [None]*4

        data = self._flowstruct_class(self.avg_data._coorddata,budget_array,index =[phystring_index,budget_index])
        
        return data
 
    def _integate_budget(self,array):
        y = self.CoordDF['y']
        if self.Domain.is_channel:
            y_mid = (array.shape[0]+1)//2
            array_half = array[:y_mid][::-1]
            y_half = y[:y_mid][::-1]
            int_array = cumtrapz(array_half,y_half,initial=0,axis=0)
            return np.concatenate([int_array[::-1],int_array[::-1]],axis=0)
        elif self.Domain.is_blayer:
            array_half = array[::-1]
            y_half = y[::-1]
            return cumtrapz( array[::-1],y[::-1],initial=0,axis=0)[::-1]
            
    
class x3d_mom_balance_z(momentum_balance_base,budgetBase):
    _flowstruct_class = fp.FlowStruct2D
    def _advection_extract(self,comp):
        
        U = self.mean_data['u']
        V = self.mean_data['v']
        U_comp = self.mean_data[comp]

        dudx = fp.Grad_calc(self.CoordDF,U_comp,'x')
        dudy = fp.Grad_calc(self.CoordDF,U_comp,'y')

        return -1.*(U*dudx + V*dudy)

    def _pressure_grad(self, comp):
        
        pressure = self.mean_data['p']
        dir = chr(ord(comp)-ord('u') + ord('x'))

        return -1.0*self.Domain.Grad_calc(self.avg_data.CoordDF,pressure,dir)

    def _viscous_extract(self,  comp):

        U_comp = self.mean_data[comp]
        dudx = fp.Grad_calc(self.CoordDF,U_comp,'x')
        d2udx2 = fp.Grad_calc(self.CoordDF,dudx,'x')
        dudy = fp.Grad_calc(self.CoordDF,U_comp,'y')
        d2udy2 = fp.Grad_calc(self.CoordDF,dudy,'y')
        REN = self.avg_data.metaDF['re']
        mu_star = 1.0
        return (mu_star/REN)*(d2udx2 + d2udy2)

    def _turb_transport(self, comp):
    
        comp_uu = comp + 'u'
        comp_uv = comp + 'v'

        if comp_uu[0] > comp_uu[1]:
            comp_uu = comp_uu[::-1]
        if comp_uv[0] > comp_uv[1]:
            comp_uv = comp_uv[::-1]
        uv = self.uu_data[comp_uv]

        uu = self.uu_data[comp_uu]
        uv = self.uu_data[comp_uv]

        duudx = fp.Grad_calc(self.CoordDF,uu,'x')
        duvdy = fp.Grad_calc(self.CoordDF,uv,'y')
        
        return -1*(duudx + duvdy)

    def plot_balance(self, x_list,PhyTime=None,budget_terms=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = check_list_vals(x_list)
        x_list = self.CoordDF.get_true_coords('x',x_list)

        budget_terms = self._check_terms(budget_terms)

        fig, ax, single_input = self._create_budget_axes(x_list,fig=fig,ax=ax,**kwargs)
        line_kw= update_line_kw(line_kw)


        for i,x_loc in enumerate(x_list):
            for comp in budget_terms:
                line_kw['label'] = self.title_with_math(comp)
                fig, ax[i] = self.budget_data.plot_line(comp,'y',x_loc,channel_half=True,
                                                    fig=fig,ax=ax[i],line_kw=line_kw)
            
            title = self.Domain.create_label(r"$x = %.3g$"%x_loc)
            ax[i].set_title(title,loc='right')

            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")


            x_label = self.Domain.create_label(r"$y$")
            ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)
            
        return fig, ax[0] if single_input else ax
    

    def plot_integrated_budget(self,x_list,budget_terms=None,PhyTime=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = check_list_vals(x_list)
        budget_terms = self._check_terms(budget_terms)

        fig, ax, single_input = self._create_budget_axes(x_list,fig,ax,**kwargs)
        line_kw= update_line_kw(line_kw)

        for i,x_loc in enumerate(x_list):
            for comp in budget_terms:
                
                line_kw['label'] = self.title_with_math(comp)
                
                budget = self.budget_data[PhyTime,comp]
                int_budget = self._integate_budget(budget)
                
                fig, ax[i] = self.budget_data.plot_line_data(int_budget,
                                                          'y',
                                                          x_loc,
                                                          time=PhyTime,
                                                          channel_half=True,
                                                          fig=fig,
                                                          ax=ax[i],
                                                          line_kw=line_kw)
            
            title = self.Domain.create_label(r"$x = %.2g$"%x_loc)
            ax[i].set_title(title,loc='right')                                              
            
            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")

            x_label = self.Domain.create_label(r"$y$")
            ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)
            
        return fig, ax[0] if single_input else ax
    
class x3d_mom_balance_xz(momentum_balance_base,budgetBase):
    _flowstruct_class = fp.FlowStruct1D
    def _advection_extract(self, comp):
        UV = self.mean_data[comp]*self.mean_data['v']

        return self.Domain.Grad_calc(self.avg_data.CoordDF,UV,'y')

    def _viscous_extract(self, comp):
        
        U_comp = self.mean_data[comp]
        dudy = fp.Grad_calc(self.CoordDF,U_comp,'y')
        d2udy2 = fp.Grad_calc(self.CoordDF,dudy,'y')
        
        REN = self.avg_data.metaDF['re']
        mu_star = 1.0
        
        return (mu_star/REN)*d2udy2

    def _turb_transport(self, comp):
        comp_uv = comp + 'v'

        if comp_uv[0] > comp_uv[1]:
            comp_uv = comp_uv[::-1]

        uv = self.uu_data[comp_uv]

        return -1*self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')

    def _pressure_grad(self, comp):
        U_mean = self.mean_data[comp]

        REN = self.avg_data.metaDF['re']
        d2u_dy2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.uu_data[comp + 'v']
        duv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')

        return -( (1/REN)*d2u_dy2 - duv_dy )
    

    def plot_balance(self, PhyTime=None,budget_terms=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.budget_data.check_outer(PhyTime,err=KeyError())
        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)
        budget_terms = self._check_terms(budget_terms)
        line_kw= update_line_kw(line_kw)

        for comp in budget_terms:
            line_kw['label'] = self.title_with_math(comp)
            fig, ax = self.budget_data.plot_line(comp,time=PhyTime,channel_half=True,
                                            fig=fig,ax=ax,line_kw=line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_integrated_budget(self,budget_terms=None,PhyTime=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.budget_data.check_outer(PhyTime,err=KeyError())
        budget_terms = self._check_terms(budget_terms)
        line_kw= update_line_kw(line_kw)

        fig, ax = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        for comp in budget_terms:
            line_kw['label'] = self.title_with_math(comp)
            budget = self.budget_data[PhyTime,comp]

            int_budget = self._integate_budget(budget)
            fig, ax = self.budget_data.plot_line_data(int_budget,
                                                    time=PhyTime,
                                                    channel_half=True,
                                                    fig=fig,
                                                    ax=ax,
                                                    line_kw=line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)


        ax.legend()

        return fig, ax    
    
class x3d_mom_balance_xzt(x3d_mom_balance_xz,budget_xzt_base):
    _flowstruct_class = fp.FlowStruct1D_time
    def _budget_extract(self,comp):
        
        transient = self._transient_extract(comp).T
        advection = self._advection_extract(comp).T
        pressure_grad = self._pressure_grad(comp).T
        viscous = self._viscous_extract(comp).T
        Reynolds_stress = self._turb_transport(comp).T
    
        array_concat = [transient,advection,pressure_grad,viscous,Reynolds_stress]

        budget_array = np.concatenate(array_concat,axis=0)
        
        comps = ['transient','advection','pressure gradient','viscous stresses','reynolds stresses']  
        
        times = self.avg_data.times
        comps = [[x]*len(times) for x in comps]        
        index = [list(times)*len(comps),list(chain(*comps))]
        
        budget_data = self._flowstruct_class(self._coorddata,
                                            budget_array,
                                            index =index)
        
        return budget_data    
    
    def _transient_extract(self,comp):
        u = self.mean_data[comp]
        times = self.avg_data.times
        return-1.*np.gradient(u,times,axis=-1)
    
    def _pressure_grad(self, comp):
        p_grad = super()._pressure_grad(comp)
        u = self.mean_data[comp]
        times = self.avg_data.times
        return p_grad + np.gradient(u,times,axis=-1)
        
    def plot_balance(self,times_list, budget_terms=None,fig=None, ax =None,line_kw=None,**kwargs):
        times_list = check_list_vals(times_list)
        
        fig, ax, single_input = self._create_budget_axes(times_list,fig,ax,**kwargs)
        for i,time in enumerate(times_list):
            fig, ax[i] = super().plot_balance(PhyTime=time,budget_terms=budget_terms,fig=fig,ax=ax[i],line_kw=line_kw)

            time_label = self.Domain.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in   handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax[0] if single_input else ax


    def plot_integrated_budget(self,times_list, budget_terms=None,fig=None, ax =None,line_kw=None,**kwargs):
        
        fig, ax, single_input = self._create_budget_axes(times_list,fig,ax,**kwargs)
        for i,time in enumerate(times_list):
            fig, ax[i] = super().plot_integrated_budget(PhyTime=time,budget_terms=budget_terms,fig=fig,ax=ax[i],line_kw=line_kw)
            ax[i].get_legend().remove()

            time_label = self.Domain.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = flip_leg_col(handles,4)
        labels = flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax[0] if single_input else ax    
    
class _FIK_base(budgetBase):
    _flowstruct_class = None
    def __init__(self,avg_data):
        
        self._get_stat_data(avg_data)

        self.budget_data = self._budget_extract()

        self._del_stat_data()
    def _integrate(self,y,x):
        if self.Domain.is_channel:
            mid = (self.NCL[1]+1)//2
            return simps(y[:mid],x[:mid],axis=0)
        else:
            return simps(y,x,axis=0)
        
    def _get_stat_data(self,avg_data):
        
        self.avg_data = avg_data
        self.mean_data = self.avg_data.mean_data
        self.uu_data = self.avg_data.uu_data


    def _del_stat_data(self):
        del self.mean_data
        del self.uu_data
        
    def _budget_extract(self):
        laminar = self._laminar_extract()
        turbulent = self._turbulent_extract()
        inertia = self._inertia_extract()

        array_concat = [laminar,turbulent,inertia]

        budget_array = np.stack(array_concat,axis=0)
        budget_index = ['laminar', 'turbulent','non-homogeneous']
        phystring_index = [None]*3

        budget = fp.datastruct(budget_array,index =[phystring_index,budget_index])

        return budget   
     
    @abstractmethod
    def _scale_vel(self,PhyTime):
        pass
    
    def _laminar_extract(self):
    
        bulk = self._scale_vel()
        REN = self.avg_data.metaDF['re']
        return 6.0/(REN*bulk)
    
    @abstractmethod
    def _turbulent_extract(self):
        pass
    
    @abstractmethod
    def _inertia_extract(self):
        pass
class _FIK_developing_base(_FIK_base):

    def _turbulent_extract(self):

        bulk = self._scale_vel()
        y_coords = self.avg_data.CoordDF['y']-1
        uv = self.avg_data.uu_data['uv']

        turbulent =    np.squeeze(y_coords[:,None]*uv)

        return 6.*self._integrate(turbulent,y_coords)/bulk**2

    @abstractmethod
    def _inertia_extract(self,PhyTime):
        pass  

    @abstractmethod
    def plot(self,budget_terms=None,plot_total=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        budget_terms = self._check_terms(budget_terms)
        
        kwargs = update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= update_line_kw(line_kw)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in budget_terms:
            budget_term = self.budget_data[comp].copy()
                
            label = self.title_with_math(comp)
            ax.cplot(xaxis_vals,budget_term,label=label,**line_kw)
        if plot_total:
            ax.cplot(xaxis_vals,np.sum(self.budget_data.values,axis=0),label="Total",**line_kw)

        return fig, ax
    
class x3d_FIK_z(_FIK_developing_base):
    def __init__(self,avg_data):
        self.avg_data = avg_data
        self.budget_data = self._budget_extract()
        
    def _scale_vel(self):
        return self.avg_data.bulk_velo_calc()
    
    def _inertia_extract(self):
        y_coords = self.avg_data.CoordDF['y']-1

        bulk = self._scale_vel()

        pressure = self.avg_data.mean_data['p']
        pressure_grad_x = self.Domain.Grad_calc(self.avg_data.CoordDF,pressure,'x')

        p_prime2 = pressure_grad_x - self._integrate(pressure_grad_x,y_coords)

        u_mean2 = self.avg_data.mean_data['u']**2
        uu = self.avg_data.uu_data['uu']
        d_UU_dx = self.Domain.Grad_calc(self.avg_data.CoordDF,
                                        u_mean2+uu,'x')
        
        UV = self.avg_data.mean_data['u']*self.avg_data.mean_data['v']
        d_uv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,UV,'y')

        REN = self.avg_data.metaDF['re']
        U_mean = self.avg_data.mean_data['u']
        d2u_dx2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'x'),'x')

        I_x = d_UU_dx + d_uv_dy - (1/REN)*d2u_dx2
        I_x_prime  = I_x -  self._integrate(I_x,y_coords)


        out = np.zeros_like(U_mean)
        for i,y in enumerate(y_coords):
            out[i] = (p_prime2 + I_x_prime)[i,:]*y**2


        return -3.0*self._integrate(out,y_coords)/(bulk**2) 
    
    def plot(self,*args,**kwargs):
        fig, ax = super().plot(*args,**kwargs)
        ax.set_xlabel(r"$x/\delta$")
        return fig, ax
    
class x3d_FIK_xzt(_FIK_developing_base):
    def __init__(self,avg_data):
        self.avg_data = avg_data
        self.budget_data = self._budget_extract()
    
    @property
    def times(self):
        self.budget_data.times
        
    def _scale_vel(self):
        return self.avg_data.bulk_velo_calc()
    

    def _inertia_extract(self):
        y_coords = self.avg_data.CoordDF['y']-1.0

        bulk = self._scale_vel()

        U_mean = self.avg_data.mean_data['u']
        times = self.avg_data._return_xaxis()

        REN = self.avg_data.metaDF['re']
        d2u_dy2 = fp.Grad_calc(self.avg_data.CoordDF,
                    fp.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.avg_data.uu_data['uv']
        duv_dy = fp.Grad_calc(self.avg_data.CoordDF,uv,'y')
        dudt = np.gradient(U_mean,times,axis=-1,edge_order=2)

        dpdx = (1/REN)*d2u_dy2 - duv_dy  - dudt
        # raise Exception("Check this pls")
        dp_prime_dx = dpdx - self._integrate((1/REN)*d2u_dy2 - duv_dy,y_coords) 

        UV = self.avg_data.mean_data['u']*self.avg_data.mean_data['v']
        I_x = fp.Grad_calc(self.avg_data.CoordDF,UV,'y')

        I_x_prime = I_x - self._integrate(I_x,y_coords)

        
        out = np.zeros(U_mean.shape)
        for i,y in enumerate(y_coords):
            out[i] = (I_x_prime + dp_prime_dx + dudt)[i]*y**2

        return -3.0*self._integrate(out,y_coords)/(bulk**2)

    def plot(self,budget_terms=None,*args,**kwargs):
        fig, ax = super().plot(budget_terms,PhyTime=None,*args,**kwargs)
        ax.set_xlabel(r"$t^*$")
        return fig, ax
    
class Cf_Renard_base(budgetBase):
    _flowstruct_class = None
    def __init__(self,avg_data,boundary_layer=True):
        self.avg_data = avg_data
        self.budget_data = self._budget_extract(boundary_layer=boundary_layer)
        
    def _get_stat_data(self,avg_data):
        
        self.avg_data = avg_data
        self.mean_data = self.avg_data.mean_data
        self.uu_data = self.avg_data.uu_data


    def _del_stat_data(self):
        del self.mean_data
        del self.uu_data
        
    def _budget_extract(self,boundary_layer=True):
        dissipation_y = self._dissipation_y_extract()
        dissipation_x = self._dissipation_x_extract()
        production_y = self._production_y_extract()
        production_x = self._production_x_extract()
        kinetic_x = self._kinetic_energy_x_extract()
        kinetic_y = self._kinetic_energy_y_extract()
        pressure = self._pressure_extract()

        if boundary_layer:
            array_concat = [dissipation_y,production_y,kinetic_x+kinetic_y]
            budget_index = ['dissipation', 'production','kinetic']
        else:
            array_concat = [dissipation_y,dissipation_x,production_y,production_x,
                            kinetic_x,kinetic_y,pressure]
            budget_index = ['dissipation y', 'dissipation x','production y',
                            'production x','kinetic x','kinetic y', 'pressure']
        budget_array = np.stack(array_concat,axis=0)
        phystring_index = [None]*len(budget_array)

        budget = fp.datastruct(budget_array,index =[phystring_index,budget_index])

        return budget 
    
    @abstractmethod
    def _scale_vel(self):
        pass
    
    def _dissipation_y_extract(self):
        u_mean = self.avg_data.mean_data['u']
        re = self.metaDF['re']
        dudy = fp.Grad_calc(self.CoordDF,u_mean,'y')
        dissipation = dudy*dudy/re
        
        y = self.CoordDF['y']
        u = self._scale_vel()
        return 2.0*simps(dissipation,y,axis=0)/(u*u*u)

    def _dissipation_x_extract(self):
        u = self._scale_vel()

        u_mean = self.avg_data.mean_data['u'] - u
        re = self.metaDF['re']
        dudx = fp.Grad_calc(self.CoordDF,u_mean,'x')
        dissipation = dudx*dudx/re
        
        y = self.CoordDF['y']
        return 2.0*simps(dissipation,y,axis=0)/(u*u*u)

    
    def _production_y_extract(self):
        u_mean = self.avg_data.mean_data['u']
        uv_mean = self.avg_data.uu_data['uv']
        dudy = fp.Grad_calc(self.CoordDF,u_mean,'y')

        production = - uv_mean*dudy
        y = self.CoordDF['y']
        u = self._scale_vel()
        
        return 2.0*simps(production,y,axis=0)/(u*u*u)

    def _production_x_extract(self):
        u = self._scale_vel()
        u_mean = self.avg_data.mean_data['u'] - u
        uu_mean = self.avg_data.uu_data['uu']
        dudx = fp.Grad_calc(self.CoordDF,u_mean,'x')

        production = - uu_mean*dudx
        y = self.CoordDF['y']
        return 2.0*simps(production,y,axis=0)/(u*u*u)

    def _kinetic_energy_y_extract(self):
        u = self._scale_vel()
        u_mean = self.avg_data.mean_data['u']
        v_mean = self.avg_data.mean_data['v']
        
        K = 0.5*u_mean*u_mean
        
        v_dKdy = v_mean*fp.Grad_calc(self.CoordDF,K,'y') \
                -u*v_mean*fp.Grad_calc(self.CoordDF,u_mean,'y')
        y = self.CoordDF['y']
        return 2.0*simps(v_dKdy,y,axis=0)/(u*u*u)

    def _kinetic_energy_x_extract(self):
        u = self._scale_vel()
        u_mean = self.avg_data.mean_data['u']
        
        K = 0.5*u_mean*u_mean
        
        u_dKdx = (u_mean-u)*fp.Grad_calc(self.CoordDF,K,'x')
        y = self.CoordDF['y']
        return 2.0*simps(u_dKdx,y,axis=0)/(u*u*u)
    
    def _pressure_extract(self):
        u = self._scale_vel()
        u_mean = self.avg_data.mean_data['u'] - u
        p_mean = self.avg_data.mean_data['p']
        u_dpdx = u_mean*fp.Grad_calc(self.CoordDF,p_mean,'x')
        y = self.CoordDF['y']
        return 2.0*simps(u_dpdx,y,axis=0)/(u*u*u)
    
    @abstractmethod
    def plot(self,budget_terms=None,plot_total=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        budget_terms = self._check_terms(budget_terms)
        
        kwargs = update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= update_line_kw(line_kw)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in budget_terms:
            budget_term = self.budget_data[comp].copy()
                
            label = self.title_with_math(comp)
            ax.cplot(xaxis_vals,budget_term,label=label,**line_kw)
        if plot_total:
            ax.cplot(xaxis_vals,np.sum(self.budget_data.values,axis=0),label="Total",**line_kw)

        return fig, ax
class x3d_Cf_Renard_z(Cf_Renard_base):
    def _scale_vel(self):
        return self.avg_data.bulk_velo_calc()
    
    def plot(self,*args,**kwargs):
        fig, ax = super().plot(*args,**kwargs)
        ax.set_xlabel(r"$x$")
        return fig, ax
    