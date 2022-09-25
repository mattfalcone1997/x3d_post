import flowpy as fp
import numpy as np
from abc import ABC, abstractmethod
from os.path import join
import json
import xml.etree.ElementTree as ET


def read_parameters(path):
    fn = join(path,'parameters.json')
    with open(fn,'r') as f:
        params = json.load(f)

    return params

def read_stat_z_file(file_path,shape,dtype='f8'):
    return np.fromfile(file_path,dtype=dtype).reshape(shape)

class _stathandler_base(ABC):
    _flowstruct_class = None
    def _check_attr(self,attr):
        if not hasattr(self,attr):
            raise AttributeError(f"Attribute {attr} must be "
                            "created to call this function")

    @abstractmethod
    def _get_data(self,path,names,it):
        pass

    def _get_nstat(self,it):
        return (it - self.metaDF['initstat']) // self.metaDF['istatcalc']

    def _extract_umean(self,path,it,it0):
        names = ['umean','vmean','wmean','pmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)


        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['u','v','w','p']
        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]

        mean_data = self._flowstruct_class(coorddata,
                                            l,
                                            index=index)
        return mean_data

    def _extract_uumean(self,path,it,it0):
        names = ['uumean','vvmean','wwmean',
                 'uvmean','uwmean','vwmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uu','vv','ww','uv','uw','vw']

        self._check_attr('mean_data')
        for i, comp in enumerate(comps):
            comp1, comp2 = comp
            u1 = self.mean_data[comp1]
            u2 = self.mean_data[comp2]

            l[i] = l[i] - u1*u2

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        uu_data =  self._flowstruct_class(coorddata,
                                          l,
                                          index=index)
        return uu_data

    def _extract_uuumean(self,path,it,it0):
        names = ['uuumean','uuvmean','uuwmean',
                 'uvvmean','uvwmean','uwwmean',
                 'vvvmean','vwwmean','vvwmean','wwwmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['uuu','uuv','uuw','uvv','uvw','uww',
                 'vvv','vvw','vww','www']

        self._check_attr('mean_data')
        self._check_attr('uu_data')
        for i, comp in enumerate(comps):
            comp_uu_12 = comp[:2]
            comp_uu_13 = comp[0] + comp[2]
            comp_uu_23 = comp[1:]

            comp_u_1 = comp[0]
            comp_u_2 = comp[1]
            comp_u_3 = comp[2]

            u1 = self.mean_data[comp_u_1]
            u2 = self.mean_data[comp_u_2]
            u3 = self.mean_data[comp_u_3]

            u1u2 = self.uu_data[comp_uu_12]
            u1u3 = self.uu_data[comp_uu_13]
            u2u3 = self.uu_data[comp_uu_23]

            l[i] = l[i] - (u1*u2*u3 + u1*u2u3 \
                        + u2*u1u3 + u3*u1u2)

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        uuu_data =  self._flowstruct_class(coorddata,
                                           l,
                                           index=index)
        return uuu_data

    def _extract_dudxmean(self,path,it,it0):
        names = ['dudxmean','dudymean','dudzmean',
                 'dvdxmean','dvdymean','dvdzmean',
                 'dwdxmean','dwdymean','dwdzmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['dudx','dudy','dudz',
                 'dvdx','dvdy','dvdz',
                 'dwdx','dwdy','dwdz']

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        dudx_mean =  self._flowstruct_class(coorddata,
                                            l,
                                            index=index)  

        return dudx_mean

    def _extract_pumean(self,path,it,it0):
        names = ['pumean','pvmean','pwmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['pu','pv','pw']

        self._check_attr('mean_data')
        p = self.mean_data['p']
        for i, comp in enumerate(comps):
            comp = comp[1]
            u = self.mean_data[comp]
            l[i] = l[i] - p*u

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        pu_data =  self._flowstruct_class(coorddata,
                                          l,
                                          index=index)
        return pu_data

    def _extract_pdudxmean(self,path,it,it0):
        names = ['pdudxmean','pdvdymean','pdwdzmean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['pdudx','pdvdy','pdwdz']

        self._check_attr('mean_data')
        self._check_attr('dudx_data')
        p = self.mean_data['p']
        for i, comp in enumerate(comps):
            comp = comp[1:]
            u = self.dudx_data[comp]
            l[i] = l[i] - p*u

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        pdudx_data =  self._flowstruct_class(coorddata,
                                             l,
                                             index=index)
        return pdudx_data

    def _extract_dudx2mean(self,path,it,it0):
        names = ['dudxdudxmean','dudxdvdxmean', 'dudxdwdxmean', 
                'dvdxdvdxmean', 'dvdxdwdxmean', 'dwdxdwdxmean',
                'dudydudymean', 'dudydvdymean', 'dudydwdymean', 
                'dvdydvdymean', 'dvdydwdymean', 'dwdydwdymean']

        l = self._get_data(path,names,it)
        if it0 is not None:
            l0 = self._get_data(path,names,it0)
            it = self._get_nstat(it)
            it0 = self._get_nstat(it0)

            l = (it*l - it0*l0) / (it - it0)

        geom = fp.GeomHandler(self.metaDF['itype'])
        coorddata = fp.AxisData(geom, self.CoordDF, coord_nd=None)

        comps = ['dudxdudx','dudxdvdx', 'dudxdwdx', 
                'dvdxdvdx', 'dvdxdwdx', 'dwdxdwdx',
                'dudydudy', 'dudydvdy', 'dudydwdy', 
                'dvdydvdy', 'dvdydwdy', 'dwdydwdy']

        self._check_attr('dudx_data')
        for i, comp in enumerate(comps):
            comp1 = comp[:4]
            comp2 = comp[4:]

            u1 = self.dudx_data[comp1]
            u2 = self.dudx_data[comp2]
            l[i] = l[i] - u1*u2

        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        dudx2_data =  self._flowstruct_class(coorddata,
                                             l,
                                             index=index)
        return dudx2_data  

class stat_z_handler(_stathandler_base,ABC):
    _flowstruct_class = fp.FlowStruct2D
    def _get_data(self,path,names,it):
        shape = (len(names), self.NCL[1], self.NCL[0])

        l = np.zeros(shape)
        for i, name in enumerate(names):
            fn = self._get_stat_file_z(path,name,it)
            l[i] = read_stat_z_file(fn,shape[1:])
        return l

class stat_xz_handler(stat_z_handler,ABC):
    _flowstruct_class = fp.FlowStruct1D
    def _get_data(self,path,names,it):
        l = super()._get_data(path,names,it)
        return l.mean(axis=-1)
      

class inst_reader(ABC):
    _reader_comps = {'u':'ux',
                     'v':'uy',
                     'w':'uz',
                     'p':'pp',}
    _default_comps = ['u','v','w','p']
    def _check_comps(self,comps):
        if comps is None:
            return self._default_comps

    def _extract_xml(self,fn):
        root =ET.parse(fn).getroot()

        shape_str = root.find('Domain/Topology').get('Dimensions')
        shape = tuple(int(s) for s in shape_str.split())

        geom = root.find('Domain/Geometry').findall('DataItem')

        geom_data = [None]*3

        data = geom[2].text.replace('\n','').split()
        geom_data[0] = np.array([np.float64(x) for x in data])

        data = geom[1].text.replace('\n','').split()
        geom_data[1] = np.array([np.float64(x) for x in data])

        data = geom[0].text.replace('\n','').split()
        geom_data[2] = np.array([np.float64(x) for x in data])

        return shape, geom_data

    def _extract_inst_xdmf(self,it,path,comps=None):
        data_folder = join(path,'data')

        self._check_comps(comps)

        xml_fn = join(data_folder,'snapshot-%s.xdmf'%str(it).zfill(7))
        shape, geom_data = self._extract_xml(xml_fn)

        print(geom_data)
        geom = fp.GeomHandler(self.metaDF['itype'])
        coords = fp.coordstruct({'x':geom_data[2],
                                 'y':geom_data[1],
                                 'z':geom_data[0]})

        coorddata = fp.AxisData(geom, coords, coord_nd=None)

        l =[]
        for comp in comps:
            c = self._reader_comps[comp]
            fn = join(data_folder,'%s-%s.bin'%(c,str(it).zfill(7)))

            data = np.fromfile(fn,dtype='f8').reshape(shape)
            l.append(data)
        time = it*self.metaDF['dt']
        index = [[time]*len(comps),comps]
        u_data = fp.FlowStruct3D(coorddata,
                                  np.array(l),
                                  index=index)

        return u_data




            