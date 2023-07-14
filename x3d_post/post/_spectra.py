from ._common import CommonData, CommonTemporalData
from ._average import (x3d_avg_xz,
                       x3d_avg_xzt,
                       x3d_avg_z)
from ._fluct import x3d_fluct_xzt
from ._data_handlers import (stathandler_base,
                             stat_xz_handler,
                             stat_z_handler)

from ..utils import get_iterations
import matplotlib as mpl
import os
import numpy as np
import flowpy as fp
import pyfftw
import json
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.interfaces.cache.enable()

from pyfftw.interfaces.numpy_fft import irfft, ifft, rfftn, irfftn, fft, rfft

from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_contour_kw,
                             update_line_kw)
_avg_xz_class = x3d_avg_xz
_avg_xzt_class = x3d_avg_xzt
_fluct_xzt_class = x3d_fluct_xzt
_avg_z_class = x3d_avg_z

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as simps

else:
    from scipy.integrate import simps
    

class spectra_base(stathandler_base,CommonData):
    
    def __init__(self,*args,from_hdf=False,**kwargs):
        
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._spectra_extract(*args,**kwargs)
            
    def _hdf_extract(self,fn,key=None):
        key = self._get_hdf_key(key)
        
        self.avg_data = self._get_avg_data.from_hdf(fn,key=key+'/avg_data')
        
        self.spectra_data = self._flowstruct_class.from_hdf(fn,key=key+'/spectra_data')
    
    def save_hdf(self,fn,mode,key=None):
        key = self._get_hdf_key(key)
        
        self.avg_data.save_hdf(fn,mode,key=key+'/avg_data')
        self.spectra_data.to_hdf(fn,'a',key=key+'/spectra_data')

    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)
    
    @property
    def Domain(self):
        return self.avg_data.Domain

    @property
    def _coorddata(self):
        return self.avg_data._coorddata

    def _get_new_data(self):
        pass
    def _get_old_data(self):
        pass
    
    @property
    def _meta_data(self):
        return self.avg_data._meta_data
    
class x3d_spectra_xz(stat_xz_handler,spectra_base):
    _flowstruct_class = fp.FlowStructND
    
    @property
    def _get_avg_data(self):
        return self._module._avg_xz_class
    
    def _get_spectra_data(self,it,path,it0=None,comps=None):
        stat_folder = os.path.join(path,'statistics')
        files = [f for f in os.listdir(stat_folder) if 'spectra_2d' in f]
        
        comps_avail = list(set([l.replace('spectra_2d','')\
                            .replace('.dat',' ').split()[0] \
                                for l in files]))
        if comps is None:
            comps = comps_avail
        else:
            comps = [comp for comp in comps if comp in comps_avail]
            if len(comps) < 1:
                raise ValueError("Invalid comps given to spectra")


        shape = (self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        lshape = (len(comps),self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        l = np.zeros(lshape,dtype='c16')

        L_z = self.CoordDF['z'][-1] - self.CoordDF['z'][0]
        L_x = self.CoordDF['x'][-1] - self.CoordDF['x'][0]

        for i,comp in enumerate(comps):
            fn = self._get_stat_file_z(path,'spectra_2d_'+comp,it)
            data = np.fromfile(fn,dtype='c16').reshape(shape)
            if it0 is not None:
                fn = stathandler_base._get_stat_file_z(path,'spectra_2d_'+comp,it0)
                l0 = np.fromfile(fn,dtype='c16').reshape(shape)
                it_ = self._get_nstat(it)
                it0_ = self._get_nstat(it0)

                data = (it_*data - it0_*l0) / (it_ - it0_)
            
            l[i] = 0.25*data*L_z*L_x/(np.pi*np.pi)
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        
        z = self.CoordDF['z']
        x = self.CoordDF['x']
        
        dx = np.diff(x)[0]
        dz = np.diff(z)[0]

        K_z = 2.*np.pi*np.fft.rfftfreq(2*(shape[0]-1),dz)
        K_x = 2.*np.pi*np.fft.fftfreq(shape[2],dx)

        comps = self._process_spectra(path,it,it0,l,comps, K_x, K_z)

        CoordDF = fp.coordstruct({'k_z':K_z,
                                  'y':self.CoordDF['y'],
                                  'k_x':K_x})

        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            noflip = ['omega_y']
            exclude = [comp for comp in comps if 'corr' in comp]
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],1,noflip=noflip, exclude=exclude)

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('k_z','y','k_x'),
                                    wall_normal_line='y',
                                    index=index)
    
    def _extract_dudxmean(self,path,it,it0):
        names = ['dudxmean','dudymean','dudzmean',
                 'dvdxmean','dvdymean','dvdzmean',
                 'dwdxmean','dwdymean','dwdzmean']

        comps = ['dudx','dudy','dudz',
                 'dvdx','dvdy','dvdz',
                 'dwdx','dwdy','dwdz']
        
        l = self._get_data(path,'dudx_mean',names,it,9,comps=comps)
        if it0 is not None:
            l0 = self._get_data(path,'dudx_mean',names,it0,9,comps=comps)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)
        
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        dudx_mean =  fp.FlowStruct1D(coorddata,
                                    l,
                                    index=index)  

        return dudx_mean


    def _process_spectra(self,path,it,it0,l,comps,kx, kz):
        dudx_mean = self._extract_dudxmean(path,it,it0)
        
        with open(os.path.join(path,'statistics.json'),'r') as f:
            stat_params = json.load(f).get("spectra_corr",None)
            
        if stat_params is not None:
            n = len(stat_params['y_locs'])
            spectra_corr_comps = ["%s_corr%d"%(c,i) for c in ['uu','vv','ww']\
                                    for i in range(1,n+1)]
        else:
            spectra_corr_comps = []
        
        for i,comp in enumerate(comps):
            if comp == 'pdudx':
                l[i] = 2*l[i]*kx
            if comp == 'pdwdz':
                l[i] = 2*l[i]*kz[:,None,None]
            if comp == 'pdvdy':
                l[i] = 2*l[i]
            if comp == 'omega_y':
                l[i] = -dudx_mean['dudy'][None,:,None]*l[i]*kz[:,None,None]*1.0j
            
            if comp in spectra_corr_comps:
                ind = int(comp[-1])-1
                yloc = stat_params['y_locs'][ind]
                new_comp = comp[:-1] + '_y' +str(yloc)
                comps[i] = new_comp
        
        return comps
                
                
    def _spectra_extract(self,it,path,it0=None,comps=None):
        self.avg_data = self._get_avg_data(it,path,it0)
        
        self.spectra_data = self._get_spectra_data(it,path,it0,comps=comps)
        

    def get_autocorrelation(self,comps=None,time=None):
        if comps is None:
            comps = self.spectra_data.inner_index

        shape = (len(comps),*self.NCL[::-1])
        l = np.zeros(shape)
        
        d_kz = np.diff(self.spectra_data.CoordDF['k_z'])[0]
        d_kx = np.diff(self.spectra_data.CoordDF['k_x'])[0]
        
        for i, comp in enumerate(comps):
            data = self.spectra_data[time,comp]
            autocorr =irfft(ifft(data,axis=2,norm='forward'),axis=0,norm='forward')

            l[i] = np.fft.ifftshift(autocorr,axes=(0,2))*d_kx*d_kz
            # mid1 = shape[1] //2 +1

            # l[i,mid1:,:,mid2:] = autocorr[mid1:,:,mid2:][::-1,:,::-1]
            # l[i,:mid1,:,:mid2] = autocorr[:mid1,:,:mid2][::-1,:,::-1]
            # l[i,:mid1,:,mid2:] = autocorr[:mid1,:,mid2:][::-1,:,::-1]
            # l[i,mid1:,:,:mid2] = autocorr[mid1:,:,:mid2][::-1,:,::-1]

        z_coords = self.CoordDF['z']
        x_coords = self.CoordDF['x']

        dx = 0.5*np.diff(x_coords)[0]
        dz = 0.5*np.diff(z_coords)[0]

        delta_z = np.linspace(-z_coords[-1]/2-dz,z_coords[-1]/2-dz,autocorr.shape[0])        
        delta_x = np.linspace(-x_coords[-1]/2-dx,x_coords[-1]/2-dx,autocorr.shape[2])        

        CoordDF = fp.coordstruct({'delta x':delta_x,
                                  'y':self.CoordDF['y'],
                                  'delta z':delta_z})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [[None]*len(comps),comps]

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('delta z','y','delta x'),
                                    wall_normal_line='y',
                                    index=index)
    
    def get_spectra_plot(self,comps=None,time=None):
        if comps == None:
            comps = self.spectra_data.inner_index
        
        comps = list(comps)
        mid_x = self.spectra_data.shape[-1]//2 +1
        
        data = self.spectra_data[time,comps]
        spectra = data.values[:,:,:,:mid_x]
        K_x = np.abs(self.spectra_data.CoordDF['k_x'][:mid_x])
        K_z = np.abs(self.spectra_data.CoordDF['k_z'])
        coorddata = self.spectra_data._coorddata.copy()
        coorddata.coord_centered['k_x'] = K_x 
        coorddata.coord_centered['k_z'] = K_z

        return self._flowstruct_class(coorddata,
                                    spectra,
                                    data_layout=('k_z','y','k_x'),
                                    wall_normal_line='y',
                                    index=data.index)
                
    def get_spectra_1d_x(self,comps=None,time=None):
        if comps == None:
            comps = self.spectra_data.inner_index
            
        spectra_data = self.get_spectra_plot(comps=comps,time=time)
        shape = (len(comps),self.NCL[1],self.NCL[0]//2+1)
        l = np.zeros(shape)
        z_coords = self.CoordDF['z']

        d_kz = 2*np.pi/(z_coords[-1]-z_coords[0])
        for i, comp in enumerate(comps):
            data = spectra_data[time,comp]
            
            l[i] = np.sum(np.real(data),axis=0)*d_kz
        
        CoordDF = fp.coordstruct({'y':self.CoordDF['y'],
                                  'k_x':spectra_data.CoordDF['k_x']})
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [ind for ind in spectra_data.index if ind[1] in comps]

        return self._flowstruct_class(coorddata,
                                l,
                                data_layout=('y','k_x'),
                                wall_normal_line='y',
                                index=index)
        
    def get_spectra_1d_z(self,comps=None,time=None):
        if comps == None:
            comps = self.spectra_data.inner_index
            
        spectra_data = self.get_spectra_plot(comps=comps,time=time)
        shape = (len(comps),self.NCL[2]//2+1,self.NCL[1])
        l = np.zeros(shape)
        
        x_coords = self.CoordDF['x']
        
        d_kx = 2*np.pi/(x_coords[-1]-x_coords[0])
        for i, comp in enumerate(comps):
            data = spectra_data[comp]
            l[i] = np.sum(np.real(data),axis=2)*d_kx
        
        CoordDF = fp.coordstruct({'k_z':spectra_data.CoordDF['k_z'],
                                  'y':spectra_data.CoordDF['y']})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [ind for ind in spectra_data.index if ind[1] in comps]

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('k_z','y'),
                                    wall_normal_line='y',
                                    index=index)
        
    def get_energy(self,comps=None):
        if comps == None:
            comps = self.spectra_data.inner_index

        d_kz = np.diff(self.spectra_data.CoordDF['k_z'])[0]
        d_kx = np.diff(self.spectra_data.CoordDF['k_x'])[0]
        
        l = np.zeros((len(comps),self.spectra_data.shape[1]))
        shape = (2*(self.spectra_data.shape[0]-1),*self.spectra_data.shape[1:])
        data = np.zeros(shape,dtype=np.complex128)
        for i, comp in enumerate(comps):
            data[:self.spectra_data.shape[0]]  = self.spectra_data[comp]
            data[self.spectra_data.shape[0]:]  = self.spectra_data[comp][1:-1][::-1]
            # l[i] = simps(simps(data,dx=d_kx,axis=2),
            #                    dx=d_kz,axis=0)
            l[i] = np.sum(np.real(data),axis=(0,2))*d_kz*d_kx
        return fp.FlowStruct1D(self.spectra_data._coorddata,
                               l,
                               index=[list(self.spectra_data.times)*len(comps),comps])

    def plot_spectra_2d(self,comp,coord,wavelength=False,premultiply=True,y_wall_units=True,time=None,contour_kw=None,fig=None,ax=None,**kwargs):
        
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords()
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        
        spectra_data = self.get_spectra_plot(comps=[comp],time=time)

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_x = spectra_data.CoordDF['k_x'][None,:]
        k_z = spectra_data.CoordDF['k_z'][:,None]

        x_transform = (lambda x: 2*np.pi/(x+x[1])) if wavelength else None
        c_transform = (lambda x: np.real(x*k_x*k_z)) if premultiply else np.real
        
        fig, qm = spectra_data.slice[:,val,:].plot_contour(comp,
                                                            rotate=True,
                                                            time=time,
                                                            transform_xdata=x_transform,
                                                            transform_ydata=x_transform,
                                                            transform_cdata=c_transform,
                                                            contour_kw=contour_kw,
                                                            fig=fig,
                                                            ax=ax)
        if wavelength:
            ax.set_xlabel(r"$\lambda_x$")
            ax.set_ylabel(r"$\lambda_z$")
        else:
            ax.set_xlabel(r"$\kappa_x$")
            ax.set_ylabel(r"$\kappa_z$")
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig, qm
    
    def plot_spectra_x(self,comp,
                       wavelength=False,
                       premultiply=True,
                       transform_xdata=None,
                       transform_ydata=None,
                       transform_cdata=None,
                       contour_kw=None,
                       time=None,
                       fig=None,
                       ax=None,**kwargs):

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        spectra1d = self.get_spectra_1d_x(comps=[comp],time=time)
        k_x = spectra1d.CoordDF['k_x']
        
        transform_xdata, transform_ydata, \
        transform_cdata =spectra1d._check_datatransforms(transform_xdata, 
                                                         transform_ydata,
                                                         transform_cdata)
            
        x_transform = (lambda x: 2*np.pi/(x+x[1])) if wavelength else None
        c_transform = (lambda x: np.real(transform_cdata(x)*k_x)) if premultiply else np.real

        fig, qm = spectra1d.plot_contour(comp,
                                        rotate=False,
                                        time=time,
                                        transform_ydata=x_transform,
                                        transform_cdata=c_transform,
                                        contour_kw=contour_kw,
                                        fig=fig,
                                        ax=ax)
        if wavelength:
            ax.set_xlabel(r"$\lambda_x$")
        else:
            ax.set_xlabel(r"$\kappa_x$")
            
        ax.set_ylabel(r"$y$")
            
        ax.set_xscale('log')
        
        return fig, qm
    
    def plot_spectra_z(self,comp,wavelength=False,premultiply=True,contour_kw=None,time=None,fig=None,ax=None,**kwargs):

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_z = self.spectra_data.CoordDF['k_z'][:,None]
        x_transform = (lambda x: 2*np.pi/(x+x[1])) if wavelength else None
        c_transform = (lambda x: np.real(x*k_z)) if premultiply else np.real

        spectra1d = self.get_spectra_1d_z(comps=[comp])
        fig, qm = spectra1d.plot_contour(comp,
                                         rotate=True,
                                        transform_xdata=x_transform,
                                        transform_cdata=c_transform,
                                        contour_kw=contour_kw,
                                        fig=fig,
                                        ax=ax)
        

        if wavelength:
            ax.set_xlabel(r"$\lambda_z$")
        else:
            ax.set_xlabel(r"$\kappa_z$")
        
        ax.set_ylabel(r"$y$")
        ax.set_xscale('log')

        return fig, qm
    
    def plot_correlation_2d(self,comp,coord,y_wall_units=True,time=None,fig=None,ax=None,norm=True,contour_kw=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords()
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        autocorr = self.get_autocorrelation(comps=[comp],time=time)
        contour_kw = update_contour_kw(contour_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, qm = autocorr.slice[:,val,:].plot_contour(comp,
                                                       time=time,
                                                       rotate=True,
                                                       transform_cdata=transform,
                                                       fig=fig,ax=ax,
                                                       contour_kw=contour_kw)
        
        ax.set_xlabel(r"$\Delta z$")
        ax.set_ylabel(r"$\Delta x$")
        return fig, qm
        
    
    def plot_correlation_x(self,comp,coord,y_wall_units=True,norm=True,line_kw=None,time=None,fig=None,ax=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords()
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        autocorr = self.get_autocorrelation(comps=[comp],time=time)
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, ax = autocorr.slice[0,val,:].plot_line(comp,
                                                    time=time,
                                                    transform_ydata=transform,
                                                    fig=fig,ax=ax,
                                                    label=r"$R_{%s}$"%comp,
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$R_{%s}$"%comp)
        return fig, ax
    
    def plot_correlation_z(self,comp,coord,y_wall_units=True,norm=True,time=None,line_kw=None,fig=None,ax=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords()
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        autocorr = self.get_autocorrelation(comps=[comp],time=time)
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, ax = autocorr.slice[:,val,0].plot_line(comp,
                                                    time=time,
                                                    transform_ydata=transform,
                                                    fig=fig,ax=ax,
                                                    label=r"$R_{%s}$"%comp,
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta z$")
        ax.set_ylabel(r"$R_{%s}$"%comp)
        return fig, ax
    
    
class x3d_spectra_xzt(x3d_spectra_xz,CommonTemporalData):
    _flowstruct_class = fp.FlowStructND_time
    
    @classmethod
    def from_window(cls,its,path,**kwargs):
        path_its = get_iterations(path)
        meta = cls._module._meta_class(path)
        
        spectra = cls(its,path,**kwargs)
        for it in its:
            t0 = meta.get_time(it)
            hwidth = kwargs.pop('hwidth')
            window_its = [i for i in path_its if \
                         abs(meta.get_time(i)-t0) < hwidth]
            
            window_its.remove(it)
            for i, itw in enumerate(window_its):
                a = (i+1)/(i+2)
                b = 1/(i+2)
                
                t = meta.get_time(itw)
                lspectra = cls(itw,path,**kwargs)
                for comp in spectra.spectra_data.inner_index:
                    spectra.spectra_data[t0,comp] = a*spectra.spectra_data[t0,comp]\
                                                    + b*lspectra.spectra_data[t,comp]
            
        return spectra
                                                        
    @property
    def _get_avg_data(self):
        return self._module._avg_xzt_class
    
    @property
    def times(self):
        return self.spectra_data.times
    
    @classmethod
    def from_instant(cls,it,path,**spectra_params):
        return cls(it,path,from_inst=True,**spectra_params)
    
    def _compute_2d_rfft(self,array:np.ndarray)->np.ndarray:
        
        return 0.5*fft(rfft(array,axis=0,norm='forward'),axis=2,norm='forward')/np.pi
    
    def _compute_spectra_2d(self,spec1: np.ndarray, spec2: np.ndarray)-> np.ndarray: 
        L_x = self.CoordDF['x'][-1] - self.CoordDF['x'][0]
        L_z = self.CoordDF['z'][-1] - self.CoordDF['z'][0]
        return spec1.conj()*spec2*L_x*L_z
    
    def _get_spectra_from_inst(self,it,path,spectra_level=1):
        fluct_data = self._module._fluct_xzt_class(it,path=path,
                                                  avg_data=self.avg_data)
        
        nspectra = {1:4,2:8}
        shape= (nspectra[spectra_level],self.NCL[2]//2+1,self.NCL[1],(self.NCL[0]))
        data = np.zeros(shape,dtype='c16')
        
        z = self.CoordDF['z']
        x = self.CoordDF['x']
        
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        k_z = 2.*np.pi*np.fft.rfftfreq(2*(shape[1]-1),dz)
        k_x = 2.*np.pi*np.fft.fftfreq(shape[3],dx)
        
        if spectra_level >= 1:
            u_hat = self._compute_2d_rfft(fluct_data.fluct_data['u'])
            v_hat = self._compute_2d_rfft(fluct_data.fluct_data['v'])
            w_hat = self._compute_2d_rfft(fluct_data.fluct_data['w'])
            
            data[0] = self._compute_spectra_2d(u_hat,u_hat)
            data[1] = self._compute_spectra_2d(v_hat,v_hat)
            data[2] = self._compute_spectra_2d(w_hat,w_hat)
            data[3] = self._compute_spectra_2d(u_hat,v_hat)
            comps = ['uu','vv','ww','uv']
        
        if spectra_level >= 2:
            p_hat = self._compute_2d_rfft(fluct_data.fluct_data['p'])
            
            dudx_hat = 1.0j*k_x[None,None,:]*u_hat
            dwdz_hat = 1.0j*k_z[:,None,None]*w_hat
            
            dvdy = fp.Grad_calc(self.avg_data.CoordDF,fluct_data.fluct_data['v'],'y')
            dvdy_hat = self._compute_2d_rfft(dvdy)
            
            data[4] = 2.*self._compute_spectra_2d(p_hat,dudx_hat)
            data[5] = 2.*self._compute_spectra_2d(p_hat,dvdy_hat)
            data[6] = 2.*self._compute_spectra_2d(p_hat,dwdz_hat)
            
            dudx_mean = self._extract_dudxmean(path,it,None)
            dwdx_hat = 1.0j*k_x[None,None,:]*w_hat
            dudz_hat = 1.0j*k_z[:,None,None]*u_hat

            omega_y_hat = dudz_hat-dwdx_hat
            omega_spec = self._compute_spectra_2d(omega_y_hat,v_hat)
            data[7] = -1.0j*dudx_mean['dudy'][None,:,None]*k_z[:,None,None]*omega_spec
            
            comps.extend(['pdudx','pdvdy','pdwdz','omega_y'])
            
        geom = fp.GeomHandler(self.metaDF['itype'])
        CoordDF = fp.coordstruct({'k_z':k_z,
                                  'y':self.CoordDF['y'],
                                  'k_x':k_x})

        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            noflip = ['omega_y']
            for i,comp in enumerate(index[1]):
                data[i] = self._apply_symmetry(comp,data[i],1,noflip=noflip)

        return self._flowstruct_class(coorddata,
                                      data,
                                      data_layout=('k_z','y','k_x'),
                                      wall_normal_line='y',
                                      index=index)

    def _spectra_extract(self, its, path,from_inst=False,**kwargs):
        
        self.avg_data = self._get_avg_data(path,its=its)
        its = fp.check_list_vals(its)
        for i, it in enumerate(its):
            if from_inst:
                spectra = self._get_spectra_from_inst(it,path,**kwargs)
            else:
                spectra = self._get_spectra_data(it,path,None)
            if i ==0:
                self.spectra_data = spectra
            else:
                self.spectra_data.concat(spectra)
    
    def plot_spectra_2d(self,comp,coord,time,*args, **kwargs):
        return super().plot_spectra_2d(comp,coord,*args,time=time,**kwargs)

    def plot_spectra_z(self,comp,time,*args, **kwargs):
        return super().plot_spectra_z(comp,*args,time=time,**kwargs)

    def plot_spectra_x(self,comp,time,*args, **kwargs):
        return super().plot_spectra_x(comp,*args,time=time,**kwargs)

    def plot_correlation_2d(self,comp,coord,time,*args, **kwargs):
        return super().plot_correlation_2d(comp,coord,*args,time=time,**kwargs)

    def plot_correlation_x(self,comp,coord,time,*args, **kwargs):
        return super().plot_correlation_x(comp,coord,*args,time=time,**kwargs)

    def plot_correlation_z(self,comp,coord,time,*args, **kwargs):
        return super().plot_correlation_z(comp,coord,*args,time=time,**kwargs)
        
class x3d_spectra_z(stat_z_handler,spectra_base):
    _flowstruct_class = fp.FlowStructND
    
    @property
    def _get_avg_data(self):
        return self._module._avg_z_class
    
    def _get_spectra_data(self,it,path,it0=None,comps=None):
        stat_folder = os.path.join(path,'statistics')
        files = [f for f in os.listdir(stat_folder) if 'spectra_z_z' in f]
        
        
        comps_avail = list(set([l.replace('spectra_z_z_','')\
                            .replace('.dat',' ').split()[0] \
                                for l in files]))
        if comps is None:
            comps = comps_avail
        else:
            comps = [comp for comp in comps if comp in comps_avail]

        if len(comps) < 1:
            raise ValueError("Invalid comps given to spectra")

        shape = (self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        lshape = (len(comps),self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        l = np.zeros(lshape,dtype='c16')

        L_z = self.CoordDF['z'][-1] - self.CoordDF['z'][0]
        for i,comp in enumerate(comps):
            fn = self._get_stat_file_z(path,'spectra_z_z_'+comp,it)
            data = np.fromfile(fn,dtype='c16').reshape(shape)

            if it0 is not None:
                fn = stathandler_base._get_stat_file_z(path,'spectra_z_z_'+comp,it0)
                l0 = np.fromfile(fn,dtype='c16').reshape(shape)
                it_ = self._get_nstat(it)
                it0_ = self._get_nstat(it0)

                data = (it_*data - it0_*l0) / (it_ - it0_)
            
            l[i] = 0.5*data*L_z/np.pi
            
        geom = fp.GeomHandler(self.metaDF['itype'])

        x = self.CoordDF['x']
        z = self.CoordDF['z']
        
        dz = np.diff(z)[0]
        K_z = 2.*np.pi*np.fft.rfftfreq(2*(shape[0]-1),dz)

        comps = self._process_spectra(path,it,it0,l,comps, K_z)                        
        CoordDF = fp.coordstruct({'k_z':K_z,
                                  'y':self.CoordDF['y'],
                                  'x':x})

        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l = self._apply_symmetry(comp,l,1)

        return self._flowstruct_class(coorddata,
                                      l,
                                      data_layout=('k_z','y','x'),
                                      wall_normal_line='y',
                                      index=index)

    def _extract_dudxmean(self,path,it,it0):
        names = ['dudxmean','dudymean','dudzmean',
                 'dvdxmean','dvdymean','dvdzmean',
                 'dwdxmean','dwdymean','dwdzmean']

        comps = ['dudx','dudy','dudz',
                 'dvdx','dvdy','dvdz',
                 'dwdx','dwdy','dwdz']
        
        l = self._get_data(path,'dudx_mean',names,it,9,comps=comps)
        if it0 is not None:
            l0 = self._get_data(path,'dudx_mean',names,it0,9,comps=comps)
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)
        
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],0)

        dudx_mean =  fp.FlowStruct2D(coorddata,
                                    l,
                                    index=index)  

        return dudx_mean

    def _process_spectra(self,path,it,it0,l,comps, kz):
        dudx_mean = self._extract_dudxmean(path,it,it0)
        
        with open(os.path.join(path,'statistics.json'),'r') as f:
            stat_params = json.load(f).get("spectra_corr",None)
            
        if stat_params is not None:
            n = len(stat_params['y_locs'])
            spectra_corr_comps = ["%s_corr%d"%(c,i) for c in ['uu','vv','ww']\
                                    for i in range(1,n+1)]
        else:
            spectra_corr_comps = []
        
        for i,comp in enumerate(comps):
            if comp == 'pdudx':
                l[i] = 2*l[i]
            if comp == 'pdwdz':
                l[i] = 2*l[i]*kz[:,None,None]
            if comp == 'pdvdy':
                l[i] = 2*l[i]
            if comp == 'omega_y':
                l[i] = -dudx_mean['dudy'][None,:,:]*l[i]*kz[:,None,None]*1.0j
            
            if comp in spectra_corr_comps:
                ind = int(comp[-1])-1
                yloc = stat_params['y_locs'][ind]
                new_comp = comp[:-1] + '_y' +str(yloc)
                comps[i] = new_comp
                
        return comps

    def _spectra_extract(self,it,path,it0=None,comps=None):
        self.avg_data = self._get_avg_data(it,path,it0)
        
        self.spectra_data = self._get_spectra_data(it,path,it0,comps=comps)
        
                
    def get_autocorrelation(self,comps=None,time=None):
        if comps is None:
            comps = self.spectra_data.inner_index

        shape = (len(comps),*self.NCL[::-1])
        l = np.zeros(shape)
        
        d_kz = np.diff(self.spectra_data.CoordDF['k_z'])[0]
        for i, comp in enumerate(comps):
            data = self.spectra_data[time,comp]
            autocorr = irfft(data,
                             axis=0,norm='forward')
            l[i] = np.fft.ifftshift(autocorr,axes=(0))*d_kz

            
        z_coords = self.CoordDF['z']
        dz = 0.5*np.diff(z_coords)[0]
        
        delta_z = np.linspace(-z_coords[-1]/2-dz,z_coords[-1]/2,autocorr.shape[0])        

        CoordDF = fp.coordstruct({'x':self.CoordDF['x'],
                                  'y':self.CoordDF['y'],
                                  'delta z':delta_z})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [[None]*len(comps),comps]
        
        return self._flowstruct_class(coorddata,
                                      l,
                                      data_layout=('delta z','y','x'),
                                      wall_normal_line='y',
                                      index=index)
        
        
    def plot_spectra_z(self,comp,x_val,wavelength=False,premultiply=True,contour_kw=None,time=None,fig=None,ax=None,**kwargs):
            
        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_z = self.spectra_data.CoordDF['k_z'][:,None]
        x_transform = (lambda x: 2*np.pi/(x+x[1])) if wavelength else None
        c_transform = (lambda x: np.real(x*(k_z))) if premultiply else np.real

        fig, qm = self.spectra_data.slice[:,:,x_val].plot_contour(comp,
                                                                rotate=True,
                                                                time=time,
                                                                transform_xdata=x_transform,
                                                                transform_cdata=c_transform,
                                                                contour_kw=contour_kw,
                                                                fig=fig,
                                                                ax=ax)
        

        if wavelength:
            ax.set_xlabel(r"$\lambda_z$")
        else:
            ax.set_xlabel(r"$\kappa_z$")
        
        ax.set_ylabel(r"$y$")
        ax.set_xscale('log')

        return fig, qm
    
    def plot_spectra_z_line(self,comp,x_val,y_val,wavelength=False,premultiply=True,line_kw=None,time=None,fig=None,ax=None,**kwargs):
        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        line_kw = update_line_kw(line_kw)

        k_z = self.spectra_data.CoordDF['k_z']
        x_transform = (lambda x: 2*np.pi/(x+x[1])) if wavelength else None
        y_transform = (lambda x: np.real(x*(k_z))) if premultiply else np.real

        fig, ax = self.spectra_data.slice[:,y_val,x_val].plot_line(comp,
                                                                time=time,
                                                                transform_xdata=x_transform,
                                                                transform_ydata=y_transform,
                                                                line_kw=line_kw,
                                                                fig=fig,
                                                                ax=ax)
        

        if wavelength:
            ax.set_xlabel(r"$\lambda_z$")
        else:
            ax.set_xlabel(r"$\kappa_z$")
        
        ax.set_xscale('log')

        return fig, ax
    
    def plot_corr_z_contour(self,comp,x_val,norm=True,contour_kw=None,show_positive=False,time=None,fig=None,ax=None,**kwargs):
        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)
        autocorr = self.get_autocorrelation(comps=[comp],time=time)

        contour_kw = update_contour_kw(contour_kw)
        
        norm_transform = (lambda x: x) if not norm else lambda x: x/np.amax(x)
        
        if not show_positive:
            vmin = np.amin(norm_transform(autocorr.slice[:,:,x_val][comp]))
            if 'levels' in contour_kw:
                if isinstance(contour_kw['levels'],int):
                    contour_kw['levels'] = np.linspace(vmin,0.,contour_kw['levels'])

        fig, qm = autocorr.slice[:,:,x_val].plot_contour(comp,
                                                      transform_cdata=norm_transform,
                                                      fig=fig,ax=ax,
                                                      rotate=True,
                                                      contour_kw=contour_kw)
        if not show_positive:   
            qm.cmap.set_over('w')

        ax.set_xlabel(r"$\Delta z$")
        ax.set_ylabel(r"$y$")
        return fig, qm
    
    def plot_correlation_z(self,comp,coord,x_val,y_wall_units=True,norm=True,line_kw=None,time=None,fig=None,ax=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords(x_val)
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        autocorr = self.get_autocorrelation(comps=[comp],time=time)
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, ax = autocorr.slice[:,val,x_val].plot_line(comp,
                                                        transform_ydata=transform,
                                                        fig=fig,ax=ax,
                                                        label=r"$R_{%s}$"%comp,
                                                        line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta z$")
        ax.set_ylabel(r"$R_{%s}$"%comp)
        return fig, ax
    
    def get_energy(self,comps=None,time=None):
        if comps == None:
            comps = self.spectra_data.inner_index

        d_kz = np.diff(self.spectra_data.CoordDF['k_z'])[0]
        l = np.zeros((len(comps),*self.spectra_data.shape[1:]))
        shape = (2*(self.spectra_data.shape[0]-1),*self.spectra_data.shape[1:])
        data = np.zeros(shape,dtype=np.complex128)

        for i, comp in enumerate(comps):
            data[:self.spectra_data.shape[0]]  = self.spectra_data[comp]
            data[self.spectra_data.shape[0]:]  = self.spectra_data[comp][1:-1][::-1]
            l[i] = np.sum(np.real(data),axis=0)*d_kz
            
        return fp.FlowStruct2D(self.spectra_data._coorddata,
                               l,
                               index=[list(self.spectra_data.times)*len(comps),comps])


    
class x3d_autocorr_x(CommonData,stathandler_base):
    _flowstruct_class = fp.FlowStructND
    def __init__(self,*args,from_hdf=False,**kwargs):
        
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._autocorr_extract(*args,**kwargs)
            
    @classmethod
    def from_hdf(cls,fn,key=None):
        return cls(fn,from_hdf=True,key=key)
                
    def _hdf_extract(self,fn,key=None):
        key = self._get_hdf_key(key)
        
        self.avg_data = self._get_avg_data.from_hdf(fn,key=key+'/avg_data')
        
        self.autocorr_data = self._flowstruct_class.from_hdf(fn,key=key+'/autocorr_data')
        
    def save_hdf(self,fn,mode,key=None):
        key = self._get_hdf_key(key)
        
        self.avg_data.save_hdf(fn,mode,key=key+'/avg_data')
        self.autocorr_data.to_hdf(fn,'a',key=key+'/autocorr_data')

    @property
    def Domain(self):
        return self.avg_data.Domain

    @property
    def _coorddata(self):
        return self.avg_data._coorddata

    def _get_new_data(self):
        pass
    def _get_old_data(self):
        pass
    
    @property
    def _meta_data(self):
        return self.avg_data._meta_data
    
    
    @property
    def _get_avg_data(self):
        return self._module._avg_z_class
    
    
    def _autocorr_extract(self,it,path,it0=None):
               
        self.avg_data = self._get_avg_data(it,path,it0)
        with open(os.path.join(path,'statistics.json'),'r') as f:
            stat_params = json.load(f).get("autocorrelation")
            
        shape = stat_params['shape']
        x_locs = np.array(stat_params['x_locs'])
        fn = self._get_stat_file_z(path,'autocorr_mean',it)
        l = np.fromfile(fn,dtype='f8').reshape(shape,order='F')

        if it0 is not None:
            fn = self._get_stat_file_z(path,'autocorr_mean',it0)
            l0 = np.fromfile(fn,dtype='f8').reshape(shape,order='F')
            it_ = self._get_nstat(it)
            it0_ = self._get_nstat(it0)

            l = (it_*l - it0_*l0) / (it_ - it0_)

        geom = fp.GeomHandler(self.metaDF['itype'])
        
        y = self.CoordDF['y']
        x_sep = self.CoordDF['x'][:shape[0]] - x_locs[0]

        comps = ['uu']

        CoordDF = fp.coordstruct({'delta x':x_sep,
                                  'y':self.CoordDF['y'],
                                  'x0':x_locs})
        
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            l = self._apply_symmetry('uu',l,1)

        self.autocorr_data = self._flowstruct_class(coorddata,
                                                    l[None],
                                                    data_layout=('delta x','y','x0'),
                                                    wall_normal_line='y',
                                                    index=index)
        
        
    def plot_autocorr(self,x0,coord,y_wall_units=True,norm=True,time=None,line_kw=None,fig=None,ax=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords(x0)
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)
        
        fig, ax = self.autocorr_data.slice[:,val,x0].plot_line('uu',
                                                    time=time,
                                                    transform_ydata=transform,
                                                    fig=fig,ax=ax,
                                                    label=r"$x=%.3g$"%x0,
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$R_{uu}$")
        return fig, ax