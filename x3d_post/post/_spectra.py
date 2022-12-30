from ._common import CommonData, CommonTemporalData
from ._average import (x3d_avg_xz,
                       x3d_avg_xzt,
                       x3d_avg_z)
from ._data_handlers import stathandler_base
import os
import numpy as np
import flowpy as fp
import pyfftw

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.interfaces.cache.enable()

from pyfftw.interfaces.numpy_fft import irfft, irfft2

from flowpy.plotting import (update_subplots_kw,
                             create_fig_ax_with_squeeze,
                             update_contour_kw,
                             update_line_kw)
_avg_xz_class = x3d_avg_xz
_avg_xzt_class = x3d_avg_xzt
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
    
class x3d_spectra_xz(spectra_base):
    _flowstruct_class = fp.FlowStructND
    
    @property
    def _get_avg_data(self):
        return self._module._avg_xz_class
    
    def _get_spectra_data(self,it,path,it0=None):
        stat_folder = os.path.join(path,'statistics')
        files = [f for f in os.listdir(stat_folder) if 'spectra_2d' in f]
        
        comps = [l.replace('spectra_2d_','').replace('.dat',' ').split()[0] for l in files]
        comps = list(set(comps))

        shape = (self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        lshape = (len(comps),self.NCL[2]//2+1,self.NCL[1],self.NCL[0]//2+1)
        l = np.zeros(lshape,dtype='c16')

        for i,comp in enumerate(comps):
            fn = self._get_stat_file_z(path,'spectra_2d_'+comp,it)
            data = np.fromfile(fn,dtype='c16').reshape(shape)

            if it0 is not None:
                fn = stathandler_base._get_stat_file_z(path,'spectra_2d_'+comp,it0)
                l0 = np.fromfile(fn,dtype='c16').reshape(shape)
                it_ = self._get_nstat(it)
                it0_ = self._get_nstat(it0)

                data = (it_*data - it0_*l0) / (it_ - it0_)
            
            l[i] = data[:,:,:lshape[-1]]
        geom = fp.GeomHandler(self.metaDF['itype'])
        
        z = self.CoordDF['z']
        x = self.CoordDF['x']
        
        L_z = z[-1] - z[0]
        L_x = x[-1] - x[0]
        
        K_z = 2.*np.arange(1,lshape[1]+1)*np.pi/L_z
        K_x = 2.*np.arange(1,lshape[3]+1)*np.pi/L_x
        
        CoordDF = fp.coordstruct({'k_z':K_z,
                                  'y':self.CoordDF['y'],
                                  'k_x':K_x})

        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l[i] = self._apply_symmetry(comp,l[i],1)

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('k_z','y','k_x'),
                                    wall_normal_line='y',
                                    index=index)
        
    def _spectra_extract(self,it,path,it0=None):
        self.avg_data = self._get_avg_data(it,path,it0)
        
        self.spectra_data = self._get_spectra_data(it,path,it0)
        

    def get_autocorrelation(self,comps=None):
        if comps is None:
            comps = self.spectra_data.inner_index

        shape = (len(comps),*self.NCL[::-1])
        l = np.zeros(shape)
        
        for i, comp in enumerate(comps):
            data = self.spectra_data[comp]
            autocorr =irfft(irfft(data,axis=2,norm='forward'),axis=0,norm='forward')
            mid1 = shape[1] //2 +1
            mid2 = shape[3] //2 +1

            l[i,mid1:,:,mid2:] = autocorr[mid1:,:,mid2:][::-1,:,::-1]
            l[i,:mid1,:,:mid2] = autocorr[:mid1,:,:mid2][::-1,:,::-1]
            l[i,:mid1,:,mid2:] = autocorr[:mid1,:,mid2:][::-1,:,::-1]
            l[i,mid1:,:,:mid2] = autocorr[mid1:,:,:mid2][::-1,:,::-1]

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
        index = [ind for ind in self.spectra_data.index if ind[1] in comps]

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('delta z','y','delta x'),
                                    wall_normal_line='y',
                                    index=index)
    
    def get_spectra_1d_x(self,comps=None):
        if comps == None:
            comps = self.spectra_data.inner_index
            
        shape = (len(comps),self.NCL[1],self.NCL[0]//2+1)
        l = np.zeros(shape)
        z_coords = self.CoordDF['z']

        d_kz = 2*np.pi/(z_coords[-1]-z_coords[0])
        for i, comp in enumerate(comps):
            data = self.spectra_data[comp]
            
            l[i] = simps(np.real(data),dx=d_kz,axis=0)
        
        CoordDF = fp.coordstruct({'y':self.CoordDF['y'],
                                  'k_x':self.spectra_data.CoordDF['k_x'][:shape[-1]]})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [ind for ind in self.spectra_data.index if ind[1] in comps]

        return self._flowstruct_class(coorddata,
                                l,
                                data_layout=('y','k_x'),
                                wall_normal_line='y',
                                index=index)
        
    def get_spectra_1d_z(self,comps=None):
        if comps == None:
            comps = self.spectra_data.inner_index
            
        shape = (len(comps),self.NCL[2]//2+1,self.NCL[1])
        l = np.zeros(shape)
        
        x_coords = self.CoordDF['x']
        
        d_kx = 2*np.pi/(x_coords[-1]-x_coords[0])
        for i, comp in enumerate(comps):
            data = self.spectra_data[comp]
            l[i] = simps(np.real(data),dx=d_kx,axis=2)
        
        CoordDF = fp.coordstruct({'k_z':self.spectra_data.CoordDF['k_z'],
                                  'y':self.CoordDF['y']})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [ind for ind in self.spectra_data.index if ind[1] in comps]

        return self._flowstruct_class(coorddata,
                                    l,
                                    data_layout=('k_z','y'),
                                    wall_normal_line='y',
                                    index=index)
        
        
    def plot_spectra_2d(self,comp,coord,wavelength=False,premultiply=True,y_wall_units=True,time=None,contour_kw=None,fig=None,ax=None,**kwargs):
        
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords()
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_x = self.spectra_data.CoordDF['k_x'][None,:]
        k_z = self.spectra_data.CoordDF['k_z'][:,None]
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        c_transform = (lambda x: np.real(x*k_x*k_z)) if premultiply else np.real
        
        fig, qm = self.spectra_data.slice[:,val,:].plot_contour(comp,
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
    
    def plot_spectra_x(self,comp,wavelength=False,premultiply=True,contour_kw=None,time=None,fig=None,ax=None,**kwargs):

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_x = self.spectra_data.CoordDF['k_x']
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        c_transform = (lambda x: np.real(x*k_x)) if premultiply else np.real

        spectra1d = self.get_spectra_1d_x(comps=[comp])
        k_x = spectra1d.CoordDF['k_x']
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        c_transform = (lambda x: np.real(x*k_x)) if premultiply else np.real

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
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
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

        autocorr = self.get_autocorrelation(comps=[comp])
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

        autocorr = self.get_autocorrelation(comps=[comp])
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, ax = autocorr.slice[0,val,:].plot_line(comp,
                                                    time=time,
                                                    transform_ydata=transform,
                                                    fig=fig,ax=ax,
                                                    label=f"$R_{comp}$",
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(f"$R_{comp}$")
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

        autocorr = self.get_autocorrelation(comps=[comp])
        line_kw = update_line_kw(line_kw)
        transform = None if not norm else lambda x: x/np.amax(x)

        fig, ax = autocorr.slice[:,val,0].plot_line(comp,
                                                    time=time,
                                                    transform_ydata=transform,
                                                    fig=fig,ax=ax,
                                                    label=f"$R_{comp}$",
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta z$")
        ax.set_ylabel(f"$R_{comp}$")
        return fig, ax
        
class x3d_spectra_xzt(x3d_spectra_xz,CommonTemporalData):
    _flowstruct_class = fp.FlowStructND_time
    @property
    def _get_avg_data(self):
        return self._module._avg_xzt_class
    
    def _spectra_extract(self, its, path):
        
        self.avg_data = self._get_avg_data(its,path,None)
        
        for i, it in enumerate(its):
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
        
class x3d_spectra_z(spectra_base,stathandler_base):
    _flowstruct_class = fp.FlowStructND
    
    @property
    def _get_avg_data(self):
        return self._module._avg_z_class
    
    def _spectra_extract(self,it,path,it0=None):
        self.avg_data = self._get_avg_data(it,path,it0)
        
        stat_folder = os.path.join(path,'statistics')
        files = [f for f in os.listdir(stat_folder) if 'spectra_z_z' in f]
        
        comps = [l.replace('spectra_z_z_','').replace('.dat',' ').split()[0] for l in files]
        comps = list(set(comps))

        shape = (self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        lshape = (len(comps),self.NCL[2]//2+1,self.NCL[1],self.NCL[0])
        l = np.zeros(lshape,dtype='c16')

        for i,comp in enumerate(comps):
            fn = self._get_stat_file_z(path,'spectra_z_z_'+comp,it)
            data = np.fromfile(fn,dtype='c16').reshape(shape)

            if it0 is not None:
                fn = stathandler_base._get_stat_file_z(path,'spectra_z_z_'+comp,it0)
                l0 = np.fromfile(fn,dtype='c16').reshape(shape)
                it_ = self._get_nstat(it)
                it0_ = self._get_nstat(it0)

                data = (it_*data - it0_*l0) / (it_ - it0_)
            
            l[i] = data[:,:,:lshape[-1]]
        geom = fp.GeomHandler(self.metaDF['itype'])
        
        z = self.CoordDF['z']
        x = self.CoordDF['x']
        
        L_z = z[-1] - z[0]
        
        K_z = 2.*np.arange(1,lshape[1]+1)*np.pi/L_z
        
        CoordDF = fp.coordstruct({'k_z':K_z,
                                  'y':self.CoordDF['y'],
                                  'x':x})

        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)

        index = self._get_index(it,comps)
        if self.Domain.is_channel:
            for i,comp in enumerate(index[1]):
                l = self._apply_symmetry(comp,l,1)

        self.spectra_data =  fp.FlowStructND(coorddata,
                                            l,
                                            data_layout=('k_z','y','x'),
                                            wall_normal_line='y',
                                            index=index)
        
    def get_autocorrelation(self,comps=None):
        if comps is None:
            comps = self.spectra_data.inner_index

        shape = (len(comps),*self.NCL[::-1])
        l = np.zeros(shape)
        
        for i, comp in enumerate(comps):
            data = self.spectra_data[comp]
            autocorr = irfft(data,
                             axis=0,norm='forward')
            mid1 = shape[1] //2
            
            l[i,mid1:,:,:] = autocorr[:mid1,:,:]
            l[i,:mid1,:,:] = autocorr[mid1:,:,:]
            
        z_coords = self.CoordDF['z']
        
        delta_z = np.linspace(-z_coords[-1]/2,z_coords[-1]/2,autocorr.shape[0])        

        CoordDF = fp.coordstruct({'x':self.CoordDF['x'],
                                  'y':self.CoordDF['y'],
                                  'delta z':delta_z})
        
        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, CoordDF, coord_nd=None)
        index = [ind for ind in self.spectra_data.index if ind[1] in comps]

        return fp.FlowStructND(coorddata,
                                l,
                                data_layout=('delta z','y','x'),
                                wall_normal_line='y',
                                index=index)
        
        
    def plot_spectra_z(self,comp,x_val,wavelength=False,premultiply=True,contour_kw=None,fig=None,ax=None,**kwargs):

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        contour_kw = update_contour_kw(contour_kw)

        k_z = self.spectra_data.CoordDF['k_z'][:,None]
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        c_transform = (lambda x: np.real(x*k_z)) if premultiply else np.real

        fig, qm = self.spectra_data.slice[:,:,x_val].plot_contour(comp,
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
    
    def plot_correlation_z(self,comp,coord,x_val,y_wall_units=True,norm=True,line_kw=None,fig=None,ax=None,**kwargs):
        if y_wall_units:
            CoordDF = self.avg_data.Wall_Coords(x_val)
            y = CoordDF.index_calc('y',coord)
        else:
            y = self.CoordDF.index_calc('y',coord)
            
        val = self.CoordDF['y'][y]
        

        kwargs = update_subplots_kw(kwargs)
        fig, ax = create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        autocorr = self.get_autocorrelation(comps=[comp])
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
        
        self.autocorr_data = self._flowstruct_class.from_hdf(fn,key=key+'/spectra_data')
        
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

        shape = self.metaDF['autocorr_shape']
        x_locs = np.array(self.metaDF['autocorr_x_locs'])
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
                                                    label=r"$R_{uu}$",
                                                    line_kw=line_kw)
        
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$R_{uu}$")
        return fig, ax