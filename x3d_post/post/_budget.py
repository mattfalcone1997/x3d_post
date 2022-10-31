import numpy as np
from ._data_handlers import (stat_z_handler, 
                             stat_xz_handler,
                             stat_xzt_handler)

from ._common import CommonData, classproperty
from ._average import x3d_avg_z, x3d_avg_xz, x3d_avg_xzt

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

    @abstractproperty
    def _get_avg_data(self,it,path,it0):
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

class ReynoldsBudget_base(ABC):
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

_avg_z_class = x3d_avg_z
class x3d_budget_z(ReynoldsBudget_base,budgetBase,stat_z_handler):
    _flowstruct_class = fp.FlowStruct2D

    @classproperty
    def _get_avg_data(self):
        return self._module._avg_z_class

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

    @classproperty
    def _get_avg_data(self):
        return self._module._avg_xz_class

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
class x3d_budget_xzt(x3d_budget_xz,stat_xzt_handler):
    _flowstruct_class = fp.FlowStruct1D_time
    @classproperty
    def _get_avg_data(self):
        return self._module._avg_xzt_class
    
    def _budget_init(self, comp, path, its=None):
            
        self.avg_data = self._get_avg_data(path,its=its)
        
        self._get_stat_data(its,path)

        self.budget_data = self._budget_extract(comp)

        self._del_stat_data()
        
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