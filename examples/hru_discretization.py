"""
==================
HRU discretization
==================
"""

import os

import ai4water
from ai4water.preprocessing.spatial_processing import MakeHRUs

ai4water_dir = os.path.dirname(os.path.dirname(ai4water.__file__))
shapefile_paths = os.path.join(ai4water_dir, 'examples', 'paper', 'shapefiles')
assert os.path.exists(shapefile_paths)
assert len(os.listdir(shapefile_paths))>=35

Soil_shp = os.path.join(shapefile_paths, 'soil.shp')

SubBasin_shp = os.path.join(shapefile_paths, 'sub_basins.shp')
slope_shp = os.path.join(shapefile_paths, 'slope.shp')

years = {2011: {'shapefile': os.path.join(shapefile_paths, 'lu2011.shp'), 'feature': 'NAME'},
         2012: {'shapefile': os.path.join(shapefile_paths, 'lu2012.shp'), 'feature': 'NAME'},
         2013: {'shapefile': os.path.join(shapefile_paths, 'lu2013.shp'), 'feature': 'NAME'},
         2014: {'shapefile': os.path.join(shapefile_paths, 'lu2014.shp'), 'feature': 'NAME'},
         # 2015:"D:\\Laos\\data\\landuse\\shapefiles\\LU2015.shp"
         }

#########################################
# unique_sub
#-------------------
# The simplest case, where the HRU is formed by subbaisns.

hru_object = MakeHRUs('unique_sub',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'},
                     )

hru_object.call(plot_hrus=True)