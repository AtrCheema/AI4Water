import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile

from dl4seq.utils.spatial_utils import get_sorted_dict, get_areas_geoms, check_shp_validity
from dl4seq.utils.spatial_utils import get_total_area, get_lu_paras, GifUtil
from dl4seq.utils.spatial_utils import find_records


M2ToAcre = 0.0002471     # meter square to Acre
COLORS = ['#CDC0B0', '#00FFFF', '#76EEC6', '#C1CDCD', '#E3CF57', '#EED5B7', '#8B7D6B', '#0000FF', '#8A2BE2', '#9C661F',
          '#FF4040', '#8A360F', '#98F5FF', '#FF9912', '#B23AEE', '#9BCD9B', '#8B8B00']


class MakeHRUs(object):
    """Distributes a given time series data for HRUs in a catchment according to the `hru_definition`.
    Currently it is supposed that only land use changes with time.
    Arguments:
        hru_definition: str, definition of HRU, determines how an HRU should be defined. The number and size of HRUs
                        depend upon this parameter.
        index: dict, index representing time variation of land use. The keys should represent unit of time and values
                     should be the path of land use shape file during that time period.
        soil_shape: str, path of soil shapefile. Only required if hru_definition contains soil.
        slope_shape: str, path of slope shapefile.
        subbasins_shape: str, path of sub-basin shape file
    """

    HRU_DEFINITIONS = [
        'unique_lu_sub',
        'unique_sub',
        'unique_lu',
        'unique_soil_sub',
        'unique_lu_soil',
        'unique_lu_soil_sub'
    ]

    def __init__(self,
                 hru_definition,
                 index,
                 soil_shape=None,
                 slope_shape=None,
                 subbasins_shape=None,
                 slope_categories=None,
                 verbosity=0
                 ):

        self.hru_definition = hru_definition
        self.index = index
        self.soil_shape = soil_shape
        self.slope_shape = slope_shape
        self.slope=slope_categories
        self.subbasins_shape=subbasins_shape
        self.hru_paras = OrderedDict()
        self.hru_geoms = OrderedDict()
        self.all_hrus = []
        self.hru_names = []
        self.verbosity = verbosity
        self.use_sub=True

        st, en = list(index.keys())[0], list(index.keys())[-1]
        # initiating yearly dataframes
        self.area = pd.DataFrame(index=pd.date_range(str(st) + '0101', str(en) + '1231', freq='12m'))
        self.curve_no = pd.DataFrame(index=pd.date_range(str(st) + '0101', str(en) + '1231', freq='12m'))
        # distance_to_outlet
        self.dist_to_out = pd.DataFrame(index=pd.date_range(str(st) + '0101', str(en) + '1231',
                                                            freq='12m'))
        # area of HRU as fraction of total catchment
        self.area_frac_cat = pd.DataFrame(index=pd.date_range(str(st) + '0101', str(en) + '1231',
                                                              freq='12m'))

    def get(self, plot_hrus=True):

        for yr, shp_file in self.index.items():

            _hru_paras, _hru_geoms = self.get_hrus(shp_file, yr)
            self.hru_paras.update(_hru_paras)

            if plot_hrus:
                self.plot_hrus(year=yr, _polygon_dict=self.hru_geoms, nrows=3, ncols=4,
                               bbox=self.slope_shape, annotate=False, save=False,
                               name=self.hru_definition)
        return

    def get_hrus(self, _lu_name, year):
        """
        lu_path: path of landuse shapefile
        :return:

        """
        lu_reader = shapefile.Reader(_lu_name)
        sub_reader = shapefile.Reader(self.subbasins_shape)
        soil_reader = shapefile.Reader(self.soil_shape)
        slope_reader = shapefile.Reader(self.slope_shape)

        lu_area_list_acre, lu_shp_geom_list = get_areas_geoms(lu_reader)
        sub_area_list_acre, sub_shp_geom_list = get_areas_geoms(sub_reader)
        soil_area_list_acre, soil_shp_geom_list = get_areas_geoms(soil_reader)
        slop_area_list_acre, slop_shp_geom_list = get_areas_geoms(slope_reader)

        if self.verbosity > 0:
            print('Checking validity of landuse shapefile')
        lu_shp_geom_list_new = check_shp_validity(lu_shp_geom_list, len(lu_reader.shapes()), name='landuse')
        soil_shp_geom_list_new = check_shp_validity(soil_shp_geom_list, len(soil_reader.shapes()), name='soil_shape')
        slop_shp_geom_list = check_shp_validity(slop_shp_geom_list, len(slope_reader.shapes()), name='slope_shape')

        if self.use_sub:
            self.sub_shp_geom_list = sub_shp_geom_list
        else:
            self.sub_shp_geom_list = slop_shp_geom_list
            no_of_subs = len(slope_reader.shapes())

        # if sum(LuAreaListAcre) > sum(SubAreaListAcre):
        #     print('Land use area is bigger than subbasin area')
        #     IntArea_ = [[None] * no_of_subs for _ in range(no_of_lus)]
        #     IntLuArea = [None] * no_of_lus
        #     for lu in range(no_of_lus):
        #         # Int = 0
        #         IntArea = 0
        #         for sub_ind in range(no_of_subs):
        #             Int = LuShpGeomListNew[lu].intersection(SubShpGeomList[sub_ind])
        #             code, lu_code = get_code(year=year, lu_feature=LuShpGeomListNew[lu], lu_feat_ind=lu,
        #             sub_feat=SubShpGeomList[sub_ind], sub_feat_ind=sub_ind, _hru_def=_hru_def)
        #             IntArea += Int.area * M2ToAcre
        #             anarea = Int.area * M2ToAcre
        #             IntArea_[lu][sub_ind] = anarea
        #         print('area of {0} is reduced from {1:10.3f} acres to {2:10.3f}'
        #               .format(lu_code, LuAreaListAcre[lu], IntArea))  # lu_code/j
        #         IntLuArea[lu] = IntArea
        #     print('New area of all landuses is  {}'.format(sum(IntLuArea)))
        # else:
        #     print('Land use area is equal to subbasin area')

        self.tot_cat_area = get_total_area(self.sub_shp_geom_list)
        a = 0
        hru_paras = OrderedDict()

        if self.hru_definition == 'unique_lu_sub':
            for j in range(len(sub_reader.shapes())):
                for lu in range(len(lu_reader.shapes())):
                    a += 1
                    code = self.get_hru_paras(year, self.sub_shp_geom_list, j, lu_shp_geom_list_new, lu,
                                              self.sub_shp_geom_list,
                                              lu_shp=self.index[year])

                    hru_paras[code] = {'yearless_key': code[2:]}

        elif self.hru_definition == 'unique_lu':
            for lu in range(len(lu_reader.shapes())):
                code = self.get_hru_paras(year, lu_shp_geom_list_new, lu)
                hru_paras[code] = {'yearless_key': code[2:]}

        elif self.hru_definition == 'unique_sub':
            for j in range(len(sub_reader.shapes())):
                # raise NotImplementedError
                code = self.get_hru_paras(year, self.sub_shp_geom_list, j)
                #         lu = SubShpGeomList[j]
                #         area = lu.area * M2ToAcre
                #         dist = SubShpGeomList[j].find_records('dist_outle', j)  # getting distance to outlet of each hru
                #         code, _ = self.get_code(year=year, sub_feat=SubShpGeomList[j], sub_feat_ind=j)
                #         lu_percent_x = area / tot_cat_area
                #         Hru_paras[code] = {'area': area, 'area_percent': lu_percent_x,
                #                             'distance_to_outlet': dist, 'yearless_key': code[2:]}
                hru_paras[code] = {'yearless_key': code[2:]}

        elif self.hru_definition == 'unique_soil_sub':
            for j in range(len(sub_reader.shapes())):
                for lu in range(len(soil_reader.shapes())):
                    code = self.get_hru_paras(year, self.sub_shp_geom_list, j, soil_shp_geom_list_new, lu)

                    hru_paras[code] = {'yearless_key': code[2:]}

        elif self.hru_definition == 'unique_lu_soil':
            for j in range(len(soil_reader.shapes())):
                for lu in range(len(lu_reader.shapes())):
                    code = self.get_hru_paras(year, soil_shp_geom_list_new, j, lu_shp_geom_list_new, lu,
                                              j)
                    hru_paras[code] = {'yearless_key': code[2:]}

        # for the case when an HRU is unique lu in unique soil but within a unique sub-basin/soil
        elif self.hru_definition == 'unique_lu_soil_sub':
            for s in range(len(sub_reader.shapes())):
                for j in range(len(soil_reader.shapes())):
                    for lu in range(len(lu_reader.shapes())):
                        code = self.get_hru_paras(year, soil_shp_geom_list_new, j, lu_shp_geom_list_new, lu,
                                                  s, lu_shp=self.index[year])
                        hru_paras[code] = {'yearless_key': code[2:]}

        self.hru_names = list(set(self.hru_names))

        return hru_paras, self.hru_geoms


    def get_hru_paras(self, year, first_shps, first_shp_idx, second_shps=None, second_shp_idx=None,
                      subbasin_idx=None, lu_shp=None):

        if second_shps is not None:
            intersection = first_shps[first_shp_idx].intersection(second_shps[second_shp_idx])
        else:
            intersection = first_shps[first_shp_idx]

        if second_shps is not None:
            if self.hru_definition == 'unique_lu_soil_sub':
                # in this case an HRU is intersection of soil and lu but inside a sub-basin
                sub = self.sub_shp_geom_list[subbasin_idx]
                intersection = sub.intersection(intersection)
                code, lu_code = self.get_code(year=year, lu_feature=second_shps[second_shp_idx],
                                              lu_feat_ind=second_shp_idx, sub_feat=first_shps[first_shp_idx],
                                              sub_feat_ind=first_shp_idx,
                                              subbasin_feat=self.sub_shp_geom_list[subbasin_idx],
                                              subbasin_idx=subbasin_idx,
                                              lu_shp=lu_shp)
            else:
                code, lu_code = self.get_code(year=year, lu_feature=second_shps[second_shp_idx],
                                              lu_feat_ind=second_shp_idx,
                                              sub_feat=first_shps[first_shp_idx],
                                              sub_feat_ind=first_shp_idx,
                                              lu_shp=lu_shp)
        else:
            if self.hru_definition == 'unique_lu':
                code = self.get_code(year=year, lu_feature=first_shps[first_shp_idx],
                                     lu_feat_ind=first_shp_idx)

            elif self.hru_definition == 'unique_sub':
                code = self.get_code(year=year, sub_feat=first_shps[first_shp_idx],
                                     sub_feat_ind=first_shp_idx)
            else:
                raise ValueError

        hru_name = code[3:]
        year = '20' + code[0:2]
        row_index = pd.to_datetime(year + '0131', format='%Y%m%d', errors='ignore')
        self.hru_names.append(hru_name)
        self.all_hrus.append(code)

        anarea = intersection.area * M2ToAcre

        # saving value of area for currnet HRU and for for current year in dictionary
        self.area.loc[row_index, hru_name] = anarea

        if second_shps is not None:
            self.hru_geoms[code] = [intersection, first_shps[first_shp_idx], second_shps[second_shp_idx]]
        else:
            self.hru_geoms[code] = [intersection, first_shps[first_shp_idx]]

        self.area_frac_cat.loc[row_index, hru_name] = anarea / self.tot_cat_area

        dist = None
        if 'sub' in self.hru_definition and self.use_sub:
            # getting distance to outlet of each hru
            sub_shp = self.sub_shp_geom_list[first_shp_idx]
            dist = find_records(self.subbasins_shape, 'dist_outle', first_shp_idx)
        self.dist_to_out.loc[row_index, hru_name] = dist

        cn = None
        if 'lu' in self.hru_definition:
            cn = get_lu_paras(code, 'cn')
        self.curve_no.loc[row_index, hru_name] = cn

        return code

    def get_code(self, year, lu_feature=None, lu_feat_ind=None, sub_feat=None, sub_feat_ind=None,
                 subbasin_feat=None, subbasin_idx=None, lu_shp=None):
        if len(str(year)) > 3:
            year = str(year)[2:]

        if self.hru_definition == 'unique_lu_sub':
            # lu_feature = feature
            ina, inb = 'LU'+str(year)+'_NAME', lu_feat_ind
            lu_code = find_records(lu_shp, ina, inb)
            if self.use_sub:
                sub_code = find_records(self.subbasins_shape, 'Subbasin', sub_feat_ind)
                sub_code = '_sub_' + str(sub_code)
            else:
                sub_code = sub_feat_ind  # sub_feat.find_records('id', sub_feat_ind)
                sub_code = '_slope_' + SLOPE[sub_code]
            return str(year) + sub_code + '_lu_' + lu_code, lu_code

        elif self.hru_definition == 'unique_sub':
            if self.use_sub:
                sub_code = sub_feat.find_records('Subbasin', sub_feat_ind)
                sub_code = '_sub_' + str(sub_code)
            else:
                sub_code = sub_feat_ind  # sub_feat.find_records('id', sub_feat_ind)
                sub_code = '_slope_' + SLOPE[sub_code]
            return str(year) + sub_code, sub_code[5:]

        elif self.hru_definition == 'unique_lu':
            ina, inb = 'LU' + str(year) + '_NAME', lu_feat_ind
            lu_code = lu_feature.find_records(ina, inb)
            return str(year) + "_lu_" + lu_code

        elif self.hru_definition == 'unique_soil_sub':
            ina, inb = 'SOIL_GROUP', lu_feat_ind
            soil_code = lu_feature.find_records(ina, inb)
            if self.use_sub:
                sub_code = sub_feat.find_records('Subbasin', sub_feat_ind)
                sub_code = '_sub_' + str(sub_code)
            else:
                sub_code = sub_feat_ind  # sub_feat.find_records('id', sub_feat_ind)
                sub_code = '_slope_' + SLOPE[sub_code]
            return str(year) + sub_code + '_soil_' + soil_code, soil_code

        elif self.hru_definition == 'unique_lu_soil':
            ina, inb = 'LU' + str(year) + '_NAME', lu_feat_ind
            lu_code = lu_feature.find_records(ina, inb)
            ina = 'SOIL_GROUP'
            soil_code = sub_feat.find_records(ina, sub_feat_ind)
            return str(year) + '_soil_' + str(soil_code) + '_lu_' + lu_code, soil_code

        elif self.hru_definition == 'unique_lu_soil_sub':
            if self.use_sub:
                sub_code = '_sub_' + str(find_records(self.subbasins_shape, 'Subbasin', subbasin_idx))
            else:
                sub_code = subbasin_idx  # sub_feat.find_records('id', sub_feat_ind)
                sub_code = '_slope_' + SLOPE[sub_code]
            ina, inb = 'LU' + str(year) + '_NAME', lu_feat_ind
            lu_code = find_records(lu_shp, ina, inb)
            ina = 'SOIL_GROUP'
            soil_code = find_records(self.soil_shape, ina, sub_feat_ind)

            return str(year) + sub_code + '_soil_' + str(soil_code) + '_lu_' + lu_code, soil_code


    def plot_hrus(self, year, bbox, _polygon_dict, annotate=False, nrows=3,
                  ncols=4, save=False, name='',
                  annotate_missing_hru=False):

        polygon_dict = OrderedDict()
        for k, v in _polygon_dict.items():
            if str(year)[2:] in k[0:3]:
                polygon_dict[k] = v

        # sorting dictionary based on keys so that it puts same HRU at same place for each year
        polygon_dict = get_sorted_dict(polygon_dict)

        figure, axs = plt.subplots(nrows, ncols)

        if isinstance(bbox, str):
            r = shapefile.Reader(bbox)
            bbox = r.bbox
            r.close()

        figure.set_figwidth(27)
        figure.set_figheight(12)
        axis_l = [item for sublist in list(axs) for item in sublist]
        # max_bbox = get_bbox_with_max_area(_polygon_dict)

        i = 0
        for key, axis in zip(polygon_dict, axis_l):
            i += 1
            ob = polygon_dict[key][0]
            # text0 = key.split('_')[4]+' in '+key.split('_')[1]  +' '+ key.split('_')[2]
            if ob.type == 'MultiPolygon':
                anfang_x = [None] * len(ob)
                for s_ob in range(len(ob)):
                    ob_ = ob[s_ob]
                    x, y = ob_.exterior.xy
                    axis.plot(x, y, color=np.random.rand(3, ), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                    axis.get_yaxis().set_visible(False)
                    axis.get_xaxis().set_visible(False)
                    if bbox is not None:
                        axis.set_ylim([bbox[1], bbox[3]])
                        axis.set_xlim([bbox[0], bbox[2]])
                    anfang_x[s_ob] = x[0]
                if annotate:
                    axis.annotate(key[3:], xy=(0.2, 0.1), xycoords='axes fraction', fontsize=18)

            elif ob.type == 'Polygon':
                x, y = polygon_dict[key][0].exterior.xy
                axis.plot(x, y, color=np.random.rand(3, ), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                axis.get_yaxis().set_visible(False)
                axis.get_xaxis().set_visible(False)
                if annotate:
                    axis.annotate(key[3:], xy=(0.2, 0.1), xycoords='axes fraction', fontsize=18)
                if bbox is not None:
                    axis.set_ylim([bbox[1], bbox[3]])
                    axis.set_xlim([bbox[0], bbox[2]])
                # axis.text(x[0], np.mean(y), text0, color='red', fontsize='12')

            else:  # for empty cases
                x, y = np.arange(0, 10), np.arange(0, 10)
                axis.plot(x, y, color='w')
                text1 = 'no ' + key.split('_')[4] + ' in sub-basin ' + key.split('_')[2]
                if annotate_missing_hru:
                    axis.text(x[0], np.mean(y), text1, color='red', fontsize=16)
                axis.get_yaxis().set_visible(False)
                axis.get_xaxis().set_visible(False)

        figure.suptitle('HRUs for year {}'.format(year), fontsize=22)
        # plt.title('HRUs for year {}'.format(year), fontsize=22)
        if save:
            name = 'hrus_{}.png'.format(year) if name is None else name + str(year)
            plt.savefig('plots/' + name)
        plt.show()

    def plot_as_ts(self, legend=False, **kwargs):
        plt.close('all')
        axis = self.area.plot(legend=legend, title='Variation of Area (acre) of HRUs with time', **kwargs)
        axis.legend(bbox_to_anchor=(1.1, 0.99))
        plt.show()
        return

    def plot_hru_evolution(self, hru_name, make_gif=False):
        """
        plots how the hru evolved during the years
        :param hru_name: str, name of hru to be plotted
        :param make_gif: bool, if True, a gif file will be created from evolution plots
        :return: plots the hru.
        """
        for yr in self.index.keys():
            y = str(yr)[2:] + '_'
            hru_name_year = y + hru_name  # hru name with year
            self.plot_hru(hru_name_year, self.soil_shape)

        if make_gif:
            gif = GifUtil(folder=os.path.join(os.getcwd(), 'plots'), contains=hru_name)
            gif.make_gif()
            gif.remove_images()
        return

    def make_gif(self):
        gif = GifUtil(initials=self.hru_definition, folder=os.path.join(os.getcwd(), 'plots'))
        gif.make_gif()
        gif.remove_images()
        return

    def plot(self, what, index=None, show_all_together=True):
        assert what in ['landuse', 'soil', 'subbasins', 'slope']
        if what == 'landuse':
            assert index
            shp_file = self.index[index]
        else:
            shp_file = getattr(self, f'{what}_shape')

        plot_shapefile(shp_file, show_all_together)
        return

    def plot_hru(self, hru_name, bbox=None, save=False):
        """
        plot only one hru from `hru_geoms`. The value of each key in hru_geoms is a list with three shapes
        :usage
          self.plot_an_hru(self.hru_names[0], bbox=True)
        """
        shape_list = self.hru_geoms[hru_name]

        figure, (axis_list) = plt.subplots(3)
        figure.set_figheight(14)

        if bbox:
            r = shapefile.Reader(bbox)
            bbox = r.bbox

        i = 0
        leg = hru_name
        for axs, ob in zip(axis_list, shape_list):
            i +=1

            if i==2: leg = hru_name.split('_')[1:2]
            elif i==3: leg = hru_name.split('_')[-1]

            if ob.type == 'MultiPolygon':
                for s_ob in range(len(ob)):
                    ob_ = ob[s_ob]
                    x, y = ob_.exterior.xy
                    axs.plot(x, y, color=COLORS[i], alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
            elif ob.type == 'Polygon':
                x, y = ob.exterior.xy
                axs.plot(x, y, color=COLORS[i], alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
            axs.set_title(leg, fontsize=14)

            if bbox:
                axs.set_ylim([bbox[1], bbox[3]])
                axs.set_xlim([bbox[0], bbox[2]])

        if save:
            plt.savefig('plots/' + hru_name + '.png')
        plt.show()
        return


if __name__=="__main__":

    SLOPE = {0: '0-13',
             1: '13-26',
             2: '26-39',
             3: '39-53'}

    Soil_shp = "D:\\Laos\\data\\landuse\\shapefiles\\soilmap.shp"

    SubBasin_shp = "D:\\Laos\\data\\landuse\\shapefiles\\subs1.shp"
    slope_shp = "D:\\Laos\\data\\landuse\\shapefiles\\slope_corrected_small.shp"

    years = {2011:"D:\\Laos\\data\\landuse\\shapefiles\\LU2011.shp",
            2012:"D:\\Laos\\data\\landuse\\shapefiles\\LU2012.shp",
            2013:"D:\\Laos\\data\\landuse\\shapefiles\\LU2013.shp",
            2014:"D:\\Laos\\data\\landuse\\shapefiles\\LU2014.shp",
            2015:"D:\\Laos\\data\\landuse\\shapefiles\\LU2015.shp"
             }
    hru_object = MakeHRUs('unique_lu_sub',
                          index=years,
                          subbasins_shape=SubBasin_shp,
                          soil_shape=Soil_shp,
                          slope_shape=slope_shp)
    hru_object.get()