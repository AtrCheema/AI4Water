
from typing import Union
from collections import OrderedDict

from .spatial_utils import find_records
from .spatial_utils import plot_shapefile
from .spatial_utils import get_total_area, GifUtil
from .spatial_utils import get_sorted_dict, get_areas_geoms, check_shp_validity
from ai4water.backend import os, np, pd, plt, mpl, shapefile, easy_mpl


mdates = mpl.dates

M2ToAcre = 0.0002471     # meter square to Acre
COLORS = ['#CDC0B0', '#00FFFF', '#76EEC6', '#C1CDCD', '#E3CF57', '#EED5B7',
          '#8B7D6B', '#0000FF', '#8A2BE2', '#9C661F', '#FF4040', '#8A360F',
          '#98F5FF', '#FF9912', '#B23AEE', '#9BCD9B', '#8B8B00']


class MakeHRUs(object):
    """
    Distributes a given time series data for HRUs in a catchment according to
    the `hru_definition`. Currently it is supposed that only land use changes
    with time.

    Example:
        >>> import os
        >>> from ai4water.preprocessing.spatial_processing import MakeHRUs
        >>> # shapefile_paths is the path where shapefiles are located. todo
        >>> SubBasin_shp = os.path.join(shapefile_paths, 'sub_basins.shp')
        >>> shapefile_paths = os.path.join(os.getcwd(), 'shapefiles')
        >>> hru_object = MakeHRUs('unique_lu_sub',
        ...     index={2011: {'shapefile': os.path.join(shapefile_paths, 'lu2011.shp'), 'feature': 'NAME'},
        ...     2012: {'shapefile': os.path.join(shapefile_paths, 'lu2012.shp'), 'feature': 'NAME'}},
        ...                  subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'}
        ...                 )
        >>> hru_object.call()
    """

    HRU_DEFINITIONS = [
        'unique_sub',
        'unique_lu',
        'unique_soil',
        'unique_slope',
        'unique_lu_sub',
        'unique_lu_soil',
        'unique_lu_slope',
        'unique_soil_sub',
        'unique_soil_slope',
        'unique_slope_sub',
        'unique_lu_soil_sub',
        'unique_lu_soil_slope',
    ]

    def __init__(
            self,
            hru_definition:str,
            index:dict,
            soil_shape: Union[dict, None] = None,
            slope_shape: Union[dict, None]=None,
            subbasins_shape: Union[None, dict] = None,
            save:bool = True,
            verbosity: int = 1
                 ):
        """
        Parameters
        ----------
            hru_definition :
                hru definition. For valid hru_definitions check `MakeHRUs.HRU_DEFINITIONS`
            index :
                dictionary defining shapefiles of landuse which change with time.
                For example in following a land use shapefile for two years is defined.
                All attributes in land use shapefiles must have the feature `NAME`.

                >>> {2011: {'shapefile': os.path.join(shapefile_paths, 'lu2011.shp'), 'feature': 'NAME'},
                ... 2012: {'shapefile': os.path.join(shapefile_paths, 'lu2012.shp'), 'feature': 'NAME'}}

            soil_shape :
                only applicable if `soil` exists in hru_definition.
                All attributes in land use soil.shp must have the feature `NAME`.

                >>> {'shapefile': os.path.join(shapefile_paths, 'soil.shp'), 'feature': 'NAME'}

            slope_shape :
                only applicable if `slope` exists in hru_definition.
                All attributes in slope.shp shapefiles must have the feature `percent`.

                >>> {'shapefile': os.path.join(shapefile_paths, 'slope.shp'), 'feature': 'percent'}

            subbasins_shape :
                only applicable if `sub` exists in hru_definition.
                All attributes in land use shapefiles must have the feature `id`.

                >>> {'shapefile': os.path.join(shapefile_paths, 'subbasins.shp'), 'feature': 'id'}

            save : bool
            verbosity : Determines verbosity.
        """

        if shapefile is None:
            raise ModuleNotFoundError(f"You must install pyshp package e.g. pip install pyshp")
        self.hru_definition = hru_definition

        assert hru_definition in self.HRU_DEFINITIONS, f"""
        invalid value for hru_definition '{hru_definition}' provided.
        Allowed values are 
        {self.HRU_DEFINITIONS}"""

        self.combinations = hru_definition.split('_')[1:]

        if len(self.combinations)<2:
            if isinstance(index, dict):
                if not all([i.__class__.__name__=='NoneType' for i in index.values()]):
                    assert all([i.__class__.__name__=='NoneType' for i in [soil_shape, slope_shape, subbasins_shape]]), f"""
                    hru consists of only one feature i.e. {self.combinations[0]}. Thus if index is provided then not
                    other shapefile must be given.
                    """

        self.index = index
        self.soil_shape = soil_shape
        self.slope_shape = slope_shape
        self.sub_shape = subbasins_shape
        self.hru_paras = OrderedDict()
        self.hru_geoms = OrderedDict()
        self.all_hrus = []
        self.hru_names = []
        self.save = save
        self.verbosity = verbosity

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

    def call(self, plot_hrus=True):
        """
        Makes the HRUs.

        Parameters
        ----------
            plot_hrus :
                If true, the exact area hrus will be plotted as well.
        """
        for _yr, shp_file in self.index.items():

            _hru_paras, _hru_geoms = self.get_hrus(shp_file, _yr)
            self.hru_paras.update(_hru_paras)

            if plot_hrus:
                self.plot_hrus(year=_yr, _polygon_dict=self.hru_geoms, nrows=3, ncols=4,
                               bbox=self.slope_shape, annotate=False,
                               name=self.hru_definition)
        return

    def get_hrus(self,
                 idx_shp,  # shapefile whose area distribution changes with time e.g land use
                 year
                 ):
        """
        lu_path: path of landuse shapefile
        :return:

        """

        if self.verbosity > 0:
            print('Checking validity of landuse shapefile')

        hru_paras = OrderedDict()
        a = 0

        if len(self.combinations) ==1:
            shp_name = self.combinations[0]
            if idx_shp is None:
                shp_file = getattr(self, f'{shp_name}_shape')['shapefile']
                feature = getattr(self, f'{shp_name}_shape')['feature']
            else:
                shp_file = idx_shp['shapefile']
                feature = idx_shp['feature']

            shp_reader = shapefile.Reader(shp_file)
            _, shp_geom_list = get_areas_geoms(shp_reader)
            if 'sub' not in self.hru_definition:
                shp_geom_list = check_shp_validity(shp_geom_list, len(shp_reader.shapes()), name=shp_name,
                                                   verbosity=self.verbosity)

            self.tot_cat_area = get_total_area(shp_geom_list)

            for shp in range(len(shp_reader.shapes())):
                code = f'{str(year)}_{shp_name}_{find_records(shp_file, feature, shp)}'
                hru_paras[code] = {'yearless_key': code[4:]}
                intersection = shp_geom_list[shp]
                self.hru_geoms[code] = [intersection, shp_geom_list[shp]]
                self._foo(code, intersection)

        if len(self.combinations) == 2:
            first_shp_name = self.combinations[0]
            second_shp_name = self.combinations[1]

            if idx_shp is None:
                first_shp_file = getattr(self, f'{first_shp_name}_shape')['shapefile']
                first_feature = getattr(self, f'{first_shp_name}_shape')['feature']
            else:
                first_shp_file = idx_shp['shapefile']
                first_feature = idx_shp['feature']

            second_shp_file = getattr(self, f'{second_shp_name}_shape')['shapefile']
            second_feature = getattr(self, f'{second_shp_name}_shape')['feature']

            first_shp_reader = shapefile.Reader(first_shp_file)
            second_shp_reader = shapefile.Reader(second_shp_file)

            _, first_shp_geom_list = get_areas_geoms(first_shp_reader)
            _, second_shp_geom_list = get_areas_geoms(second_shp_reader)

            if 'sub' not in self.hru_definition:
                second_shp_geom_list = check_shp_validity(second_shp_geom_list, len(second_shp_reader.shapes()),
                                                          name=second_shp_name, verbosity=self.verbosity)
                self.tot_cat_area = get_total_area(first_shp_geom_list)
            else:
                self.tot_cat_area = get_total_area(second_shp_geom_list)

            for j in range(len(second_shp_reader.shapes())):
                for lu in range(len(first_shp_reader.shapes())):
                    a += 1

                    intersection = second_shp_geom_list[j].intersection(first_shp_geom_list[lu])

                    lu_code = find_records(first_shp_file, first_feature, lu)
                    sub_code = find_records(second_shp_file, second_feature, j)
                    sub_code = f'_{second_shp_name}_' + str(sub_code)
                    code = str(year) + sub_code + f'_{first_shp_name}_' + lu_code #, lu_code

                    self.hru_geoms[code] = [intersection, second_shp_geom_list[j], first_shp_geom_list[lu]]

                    hru_paras[code] = {'yearless_key': code[4:]}
                    self._foo(code, intersection)

        if len(self.combinations) == 3:
            first_shp_name = self.combinations[0]
            second_shp_name = self.combinations[1]
            third_shp_name = self.combinations[2]

            if idx_shp is None:
                first_shp_file = getattr(self, f'{first_shp_name}_shape')['shapefile']
                first_feature = getattr(self, f'{first_shp_name}_shape')['feature']
            else:
                first_shp_file = idx_shp['shapefile']
                first_feature = idx_shp['feature']

            second_shp_file = getattr(self, f'{second_shp_name}_shape')['shapefile']
            second_feature = getattr(self, f'{second_shp_name}_shape')['feature']
            third_shp_file = getattr(self, f'{third_shp_name}_shape')['shapefile']
            third_feature = getattr(self, f'{third_shp_name}_shape')['feature']

            first_shp_reader = shapefile.Reader(first_shp_file)
            second_shp_reader = shapefile.Reader(second_shp_file)
            third_shp_reader = shapefile.Reader(third_shp_file)

            _, first_shp_geom_list = get_areas_geoms(first_shp_reader)
            _, second_shp_geom_list = get_areas_geoms(second_shp_reader)
            _, third_shp_geom_list = get_areas_geoms(third_shp_reader)

            if 'sub' not in self.hru_definition:  # todo
                second_shp_geom_list = check_shp_validity(second_shp_geom_list, len(second_shp_reader.shapes()),
                                                          name=second_shp_name,
                                                          verbosity=self.verbosity)
                self.tot_cat_area = get_total_area(first_shp_geom_list)
            else:
                self.tot_cat_area = get_total_area(second_shp_geom_list)

            for s in range(len(third_shp_reader.shapes())):
                for j in range(len(second_shp_reader.shapes())):
                    for lu in range(len(first_shp_reader.shapes())):

                        intersection = second_shp_geom_list[j].intersection(first_shp_geom_list[lu])

                        sub = third_shp_geom_list[s]
                        intersection = sub.intersection(intersection)

                        sub_code = f'_{third_shp_name}_' + str(find_records(third_shp_file, third_feature, s))
                        lu_code = find_records(first_shp_file, first_feature, lu)
                        soil_code = find_records(second_shp_file, second_feature, j)
                        code = str(year) + sub_code + f'_{second_shp_name}_' + str(soil_code) + f'_{first_shp_name}_' + lu_code

                        self.hru_geoms[code] = [intersection, second_shp_geom_list[j], first_shp_geom_list[lu]]

                        hru_paras[code] = {'yearless_key': code[4:]}
                        self._foo(code, intersection)

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

        self.hru_names = list(set(self.hru_names))

        return hru_paras, self.hru_geoms

    def _foo(self, code, intersection):
        hru_name = code[5:]
        year = code[0:4]
        row_index = pd.to_datetime(year + '0131', format='%Y%m%d', errors='ignore')
        self.hru_names.append(hru_name)
        self.all_hrus.append(code)
        anarea = intersection.area * M2ToAcre

        # saving value of area for currnet HRU and for for current year in dictionary
        self.area.loc[row_index, hru_name] = anarea
        self.area_frac_cat.loc[row_index, hru_name] = anarea / self.tot_cat_area
        return

    def plot_hrus(self, year, bbox, _polygon_dict, annotate=False, nrows=3,
                  ncols=4, name='',
                  annotate_missing_hru=False):

        polygon_dict = OrderedDict()
        for k, v in _polygon_dict.items():
            if str(year) in k[0:4]:
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
                text1 = 'no ' + key.split('_')[1] + ' in sub-basin ' + key.split('_')[2]
                if annotate_missing_hru:
                    axis.text(x[0], np.mean(y), text1, color='red', fontsize=16)
                axis.get_yaxis().set_visible(False)
                axis.get_xaxis().set_visible(False)

        figure.suptitle('HRUs for year {}'.format(year), fontsize=22)
        # plt.title('HRUs for year {}'.format(year), fontsize=22)
        if self.save:
            name = 'hrus_{}.png'.format(year) if name is None else name + str(year)
            plt.savefig('plots/' + name)
        plt.show()

    def plot_as_ts(self, name=None, show=True, **kwargs):
        """hru_object.plot_as_ts(save=True, min_xticks=3, max_xticks=4"""

        figsize = kwargs.get('figsize', (12, 6))
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        tick_fs = kwargs.get('tick_fs', 14)
        axis_label_fs = kwargs.get('axis_label_fs', 18)
        bbox_to_anchor = kwargs.get('bbox_to_anchor', (1.1, 0.99))
        leg_fs = kwargs.get('leg_fs', 14)
        markerscale = kwargs.get('markerscale', 2)
        style=  kwargs.get('style', '-s')
        title = kwargs.get('title', False)
        min_xticks = kwargs.get('min_xticks', None)
        max_xticks = kwargs.get('max_xticks', None)

        plt.close('all')
        _, axis = plt.subplots(figsize=figsize)
        for area in self.area:
            axis.plot(self.area[area], style, label=area)
        axis.legend(fontsize=leg_fs, markerscale=markerscale, bbox_to_anchor=bbox_to_anchor)
        axis.set_xlabel('Time', fontsize=axis_label_fs)
        axis.set_ylabel('Area (Acres)', fontsize=axis_label_fs)
        axis.tick_params(axis="x", which='major', labelsize=tick_fs)
        axis.tick_params(axis="y", which='major', labelsize=tick_fs)

        if min_xticks is not None:
            assert isinstance(min_xticks, int)
            assert isinstance(max_xticks, int)
            loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
            axis.xaxis.set_major_locator(loc)
            fmt = mdates.AutoDateFormatter(loc)
            axis.xaxis.set_major_formatter(fmt)
        if title:
            plt.suptitle('Variation of Area (acre) of HRUs with time')
        if self.save:
            if name is None:
                name = self.hru_definition
            plt.savefig(f'{name}_hru_as_ts.png', dpi=300, bbox_inches=bbox_inches)

        if show:
            plt.show()

        return

    def plot_hru_evolution(self, hru_name, make_gif=False):
        """
        plots how the hru evolved during the years

        Parameters
        ----------
            hru_name : str,
                name of hru to be plotted
            make_gif : bool
                if True, a gif file will be created from evolution plots

        Returns
        -------
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

    def plot_hru(self, hru_name, bbox=None):
        """
        plot only one hru from `hru_geoms`.
        The value of each key in hru_geoms is a list with three shapes

        Examples
        --------
          >>> self.plot_an_hru(self.hru_names[0], bbox=True)
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

        if self.save:
            plt.savefig('plots/' + hru_name + '.png')
        plt.show()
        return

    def draw_pie(self,
                 year:int,
                 n_merge:int=0,
                 title:bool=False,
                 name:str=None,
                 show:bool = True,
                 **kwargs):
        """
        todo draw nested pie chart for all years
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
        Draws a pie chart showing relative area of HRUs for a particular year.
        Since the hrus can change with time, selecting one year is based on supposition
        that area of hrus remain constant during the whole year.

        Parameters
        ----------
            year : int,
                the year for which area of hrus will be used.
            n_merge :
                number of hrus to merge
            title :
            name :
            show :
            kwargs :
                Following keyword arguments are allowed
                shadow
                strartangle
                autopct
                textprops
        """
        shadow = kwargs.get('shadow', True)
        startangle = kwargs.get('startangle', 90)
        autopct = kwargs.get('autopct', '%1.1f%%')
        textprops = kwargs.get('textprops', {})

        idx = str(year) + '-01-31'
        area_unsort = self.area.loc[idx]
        area = area_unsort.sort_values()
        merged = area[area.index[0]:area.index[n_merge-1]]
        rest = area[area.index[n_merge]:]

        if n_merge==0:
            assert len(merged) == len(area)
            merged_vals= []
            merged_labels =  []
        else:
            merged_vals = [sum(merged.values)]
            merged_labels = ['{} HRUs'.format(n_merge)]

        vals = list(rest.values) + merged_vals
        labels = list(rest.keys()) + merged_labels

        explode = [0 for _ in range(len(vals))]
        explode[-1] = 0.1

        labels_n = []
        for l in labels:
            labels_n.append(l.replace('lu_', ''))

        if title:
            title = 'Areas of HRUs for year {}'.format(year)

        if name is None: name = self.hru_definition
        name = f'{len(self.hru_names)}hrus_for_{year}_{name}.png'

        return easy_mpl.pie(fractions=vals,
                   labels=labels_n,
                   explode=tuple(explode),
                   autopct=autopct, shadow=shadow, startangle=startangle, textprops=textprops,
                   title=title, name=name, save=self.save, show=show)
