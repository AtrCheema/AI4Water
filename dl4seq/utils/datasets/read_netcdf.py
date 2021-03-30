import os
import netCDF4


for _f in os.listdir('datasets\\HYSETS'):

    if _f.endswith('.nc'):

        nc = netCDF4.Dataset(os.path.join(os.getcwd(), f'datasets\\HYSETS\\{_f}'))

        print(f'\nreading file {_f} ')
        for var in nc.variables:
            print(var, nc[var].shape)

        nc.close()