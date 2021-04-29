import os
import netCDF4


for _f in os.listdir('data\\HYSETS'):

    if _f.endswith('.nc'):

        nc = netCDF4.Dataset(os.path.join(os.getcwd(), f'data\\HYSETS\\{_f}'))

        print(f'\nreading file {_f} ')
        for var in nc.variables:
            print('{0:30} {1}'.format(var, nc[var].shape))

        nc.close()