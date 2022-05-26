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

#########################################

hru_object.call(plot_hrus=True)

#########################################

for yr in years:
    hru_object.draw_pie(yr, title=False, n_merge=0, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_soil
#-------------------

hru_object = MakeHRUs('unique_soil',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'}
                     )

#########################################

hru_object.call(plot_hrus=True)

#########################################

for yr in years:
    hru_object.draw_pie(yr, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_soil
#-------------------

#########################################
# unique_lu
#-------------------
# Since the land use varies with time, we will include it in index.

hru_object = MakeHRUs('unique_lu',
                      index=years,
                     )

#########################################

hru_object.call(plot_hrus=True)

#########################################

for yr in years:
    hru_object.draw_pie(yr, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_slope
#------------------

hru_object = MakeHRUs('unique_slope',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      slope_shape={'shapefile': slope_shp, 'feature': 'percent'}
                     )

#########################################

hru_object.call(plot_hrus=False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_lu_sub
#------------------

#########################################

hru_object = MakeHRUs('unique_lu_sub',
                      index=years,
                      subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'}
                     )

#########################################

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=12, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_lu_soil
#------------------
# combination of land use and soil

hru_object = MakeHRUs('unique_lu_soil',
                      index=years,
                      soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'}
                     )

#########################################

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=4, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_lu_slope
#------------------

hru_object = MakeHRUs('unique_lu_slope',
                      index=years,
                      slope_shape={'shapefile': slope_shp, 'feature': 'percent'}
                     )

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=7, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_soil_sub
#------------------

hru_object = MakeHRUs('unique_soil_sub',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'},
                      soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'}
                     )

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=7, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_soil_slope
#-------------------

hru_object = MakeHRUs('unique_soil_slope',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      slope_shape={'shapefile': slope_shp, 'feature': 'percent'},
                      soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'}
                     )

#########################################

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=3, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_slope_sub
#-------------------

hru_object = MakeHRUs('unique_slope_sub',
                      index={2011: None, 2012: None, 2013:None, 2014:None},
                      slope_shape={'shapefile': slope_shp, 'feature': 'percent'},
                      subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'}
                     )

#########################################

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=7, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)

#########################################
# unique_lu_soil_slope
#----------------------

hru_object = MakeHRUs('unique_lu_soil_slope',
                      index=years,
                      slope_shape={'shapefile': slope_shp, 'feature': 'percent'},
                      soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'},
                     )

#########################################

hru_object.call(False)

#########################################

for yr in years:
    hru_object.draw_pie(yr, n_merge=29, title=False, save=True, textprops={'fontsize': '12'})

#########################################

hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=True)
