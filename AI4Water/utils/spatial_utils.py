import os
import random
from collections import OrderedDict

import imageio
try:
    from shapely.geometry import shape
except FileNotFoundError:
    raise FileNotFoundError(f"""
If you installed shapely using pip, try to resintall it 
(after uninstalling the previous installtin obviously)
by manually downloading the wheel file from 
https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
and then istalling using the wheel file using following command
pip install path/to/wheel.whl""")
import shapefile
import matplotlib.pyplot as plt
import numpy as np

M2ToAcre = 0.0002471     # meter square to Acre

COLORS = ['#CDC0B0', '#00FFFF', '#76EEC6', '#C1CDCD', '#E3CF57', '#EED5B7', '#8B7D6B', '#0000FF', '#8A2BE2', '#9C661F',
          '#FF4040', '#8A360F', '#98F5FF', '#FF9912', '#B23AEE', '#9BCD9B', '#8B8B00']



def get_sorted_dict(dictionary):
    """sorts the dictionary based on its keys and returns the new dictionary named sorted_dict"""
    sorted_dict = OrderedDict()
    for k in sorted(dictionary):
        sorted_dict[k] = dictionary[k]
    return sorted_dict


def get_areas_geoms(shp_reader):
    """returns lists containing areas of all records in shapefile and geometries of all records
     in shape file and number of records in shapefile"""
    shapes = shp_reader.shapes()

    geometries = [None] * len(shapes)  # a container for geometries of shapefile
    areas = [None] * len(shapes)  # a container for areas of shapefile
    for shp in range(len(shapes)):
        feature = shp_reader.shapeRecords()[shp]  # pyshp
        first = feature.shape.__geo_interface__  # pyshp
        geometries[shp] = shape(first)  # pyshp to shapely  geometry
        areas[shp] = shape(first).area * M2ToAcre

    return areas, geometries


def check_shp_validity(geom_list, no_of_lus, name='landuse'):
    new_geom_list = [None] * no_of_lus  # a container for geometries of landuse shapefile with corrected topology
    for lu in range(no_of_lus):  # iterating over each landuse
        luf = geom_list[lu]
        if not luf.is_valid:
            n = 0
            for j in range(len(luf)):  # iterating over each ring of feature which is invalid
                if not luf[j].is_valid:  # checking which ring in feature is invalid
                    n = n + 1  # counting number of invalid rings in a feature
            new_geom_list[lu] = luf.buffer(0)  # correcting the 'self-intersection' for the feature which is invalid
        else:
            new_geom_list[lu] = luf

    # checking the validity of each landuse once again to make sure that each landuse's is valid
    for lu in range(no_of_lus):
        sub_lu = new_geom_list[lu]
        if sub_lu.is_valid:
            print('{} {} is valid now'.format(name, lu))
        else:
            print('{} {} is still invalid'.format(name, lu))
    return new_geom_list


def get_total_area(file_to_read):
    shape_area = 0.0
    for sub_shp in file_to_read:
        shape_area += sub_shp.area*M2ToAcre
    return shape_area


def get_lu_paras(code, para):
    """get some landuse related constand parameters"""
    val = None
    dict_ = {'Urban': {'cn': 58, 'albedo': 16},
             'Forest': {'cn': 77, 'albedo': 16},
             'Teak': {'cn': 82, 'albedo': 18},
             'Crop': {'cn': 80, 'albedo': 23}}
    for key in dict_:
        if key in code:
            val = dict_[key][para]

    if val is None:
        raise ValueError('value of curve number cannot be none, wrong landuse provided.')
    return val


class GifUtil(object):

    def __init__(self, folder,initials=None, contains=None):
        self.init = initials  # starting name of files
        self.contains = contains
        self.input_files = []   # container for input files
        self.get_all_files(initials)   #get files to make gif
        self.dir = folder  # folder containing input files and output gif

    def get_all_files(self, init):
        for file in os.listdir("./plots"):
            if file.endswith(".png"):
                if self.init:
                    if file.startswith(init):
                        self.input_files.append(file)
                if self.contains:
                    if self.contains in file:
                        self.input_files.append(file)

    def make_gif(self, duration=0.5, name=None):
        if name is None:
            if self.init:
                name = self.init
            else:
                name = self.contains

        images = []
        for file in np.sort(self.input_files):
            filename = os.path.join(self.dir, file)
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(self.dir, name + '.gif'), images, duration=duration)

    def remove_images(self):
        for img in self.input_files:
            path = os.path.join(self.dir, img)
            if os.path.exists(path):
                os.remove(path)


def find_records(shp_file, record_name, feature_number):
    """find the metadata about feature given its feature number and column_name which contains the data"""
    shp_reader = shapefile.Reader(shp_file)
    col_no = find_col_name(shp_reader, record_name)

    if col_no == -99:
        raise ValueError(f'no column named {record_name} found in {shp_reader.shapeName}')
    else:
        # print(col_no, 'is the col no')
        name = get_record_in_col(shp_reader, feature_number, col_no)
    return name


def find_col_name(shp_reader, field_name):
    _col_no = 0
    col_no = -99
    for fields in shp_reader.fields:
        _col_no +=1
        for field in fields:
            if field == field_name:
                col_no = _col_no
                break
    return col_no


def get_record_in_col(shp_reader, i, col_no):
    recs = shp_reader.records()
    col_no = col_no - 2  #-2, 1 for index reduction, 1 for a junk column shows up in records
    return recs[i][col_no]


def plot_shapefile(shp_files,
                   labels=None,
                   show_all_together=True,
                   bbox_shp=None,
                   recs=None, rec_idx=None,
                   leg_kws=None,
                   save=False,
                   colors=None,
                   markersize=12,
                   save_kws=None):

    """
    leg_kws:{'bbox_to_anchor': (1.02, -0.15),
                   'numpoints': 1,
                   'fontsize': 16,
                   'markerscale':2}
    save_kws:{'fname': 'point_plot', 'bbox_inches': 'tight'}
    """
    if not isinstance(shp_files, list):
        shp_files = [shp_files]

    if leg_kws is None:
        leg_kws = {'bbox_to_anchor': (0.93, -0.15),
                   'numpoints': 1,
                   'fontsize': 16,
                   'markerscale':2}
    if labels is None:
        labels = {}
    if save_kws is None:
        save_kws = {'fname': 'point_plot', 'dpi': 300, 'bbox_inches': 'tight'}

    records = shapefile.Reader(shp_files[0]).shapeRecords()
    Colors = random.choices(COLORS, k=len(records))

    if len(shp_files) > 1:
        for i in range(1, len(shp_files)):
            shp_reader = shapefile.Reader(shp_files[i])
            records += shp_reader.shapeRecords()
            Colors += random.choices(COLORS, k=len(shp_reader.shapeRecords()))

    plt.close('all')
    for feature, n in zip(records, Colors):

        if recs is not None:
            assert isinstance(rec_idx, int)
            rec = feature.record[rec_idx]
        else:
            rec, recs = '', ''

        if rec in recs:
            f_if = feature.shape.__geo_interface__
            if f_if is None:
                pass
            else:
                if f_if['type'].lower() in ['point']:  # it is point
                    c = colors.get(rec, random.choice(COLORS))
                    plt.plot(*f_if['coordinates'], '*', label=labels.get(rec, rec), color=c, markersize=markersize)
                else:
                    plot_polygon_feature(feature, n, shapefile.Reader(shp_files[0]).bbox)

    if bbox_shp is not None:
        shp_reader = shapefile.Reader(bbox_shp)
        records = shp_reader.shapeRecords()
        for feature, n in zip(records, Colors):
            plot_polygon_feature(feature, n, shapefile.Reader(shp_files[0]).bbox)

        plt.legend(**leg_kws)
        if not show_all_together:
            plt.show()

    if save:
        plt.savefig(**save_kws)
    # if show_all_together:
    plt.show()
    #shp_reader.close()
    return


def plot_polygon_feature(feature, n, bbox):
    f_if = feature.shape.__geo_interface__
    polys = len(f_if['coordinates'])
    def_col = n
    for i in range(polys):
        a = np.array(f_if['coordinates'][i])
        if a.ndim < 2 and len(a.shape) > 0:
            c = a
            m = max([len(ci) for ci in c])
            for ci in c:
                col = 'k' if len(ci) != m else def_col
                x = np.array([k[0] for k in ci])
                y = np.array([k[1] for k in ci])
                plt.plot(x, y, col, label="__none__", linewidth=0.5)

        elif len(a.shape) > 0:
            b = a.reshape(-1, 2)
            plt.plot(b[:, 0], b[:, 1], def_col)
        plt.ylim([bbox[1], bbox[3]])
        plt.xlim([bbox[0], bbox[2]])
    return