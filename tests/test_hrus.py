import os

from ai4water.pre_processing.spatial_processing import MakeHRUs


SLOPE = {0: '0-13',
         1: '13-26',
         2: '26-39',
         #3: '39-53'
         }

shapefile_paths = os.path.join(os.path.dirname(os.getcwd()), 'examples', 'paper_figs', 'shapefiles')
Soil_shp = os.path.join(shapefile_paths, 'soil.shp')

SubBasin_shp = os.path.join(shapefile_paths, 'sub_basins.shp')
slope_shp = os.path.join(shapefile_paths, 'slope.shp')

years = {2011: {'shapefile': os.path.join(shapefile_paths, 'lu2011.shp'), 'feature': 'NAME'},
         2012: {'shapefile': os.path.join(shapefile_paths, 'lu2012.shp'), 'feature': 'NAME'},
         2013: {'shapefile': os.path.join(shapefile_paths, 'lu2013.shp'), 'feature': 'NAME'},
         2014: {'shapefile': os.path.join(shapefile_paths, 'lu2014.shp'), 'feature': 'NAME'},
         # 2015:"D:\\Laos\\data\\landuse\\shapefiles\\LU2015.shp"
         }

years_none = {2011: None, 2012: None, 2013:None, 2014:None}

hru_definitions = {
    'unique_sub': 0,
    'unique_soil': 0,
    'unique_lu': 0,
    'unique_slope': 0,
    'unique_lu_sub': 0,
    'unique_lu_soil': 0,
    'unique_lu_slope': 0,
    'unique_soil_sub': 0,
    'unique_soil_slope': 0,
    'unique_slope_sub': 0,
    'unique_lu_soil_sub': 50,
    'unique_lu_soil_slope': 30,
}

for hru_def, n_merges in hru_definitions.items():
    print(f"{'*'*10}{hru_def}{'*'*10}")
    hru_object = MakeHRUs(hru_def,
                          index=years if 'lu' in hru_def else years_none,
                          subbasins_shape={'shapefile': SubBasin_shp, 'feature': 'id'} if 'sub' in hru_def else None,
                          soil_shape={'shapefile': Soil_shp, 'feature': 'NAME'} if 'soil' in hru_def else None,
                          slope_shape={'shapefile': slope_shp, 'feature': 'percent'} if 'slope' in hru_def else None
                          )

    hru_object.call(plot_hrus=False if 'slope' in hru_def else False)

    for yr in years:
        hru_object.draw_pie(yr, title=False, n_merge=n_merges, save=False, textprops={'fontsize': '12'})

    hru_object.plot_as_ts(min_xticks=3, max_xticks=4, save=False)