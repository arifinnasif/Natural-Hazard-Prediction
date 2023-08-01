import netCDF4
import os
import datetime
import numpy as np

cwd = os.getcwd()

lat_min = 29
lat_max = 31

lon_min = -88
lon_max = -86

dim = 50



file2read = netCDF4.Dataset(cwd+'\data.nc', 'r')

dt_start = datetime.datetime.fromtimestamp(float(file2read.variables['product_time_bounds'][0]))
dt_end = datetime.datetime.fromtimestamp(float(file2read.variables['product_time_bounds'][1]))

print(dt_start, dt_end)

flash_count = file2read.variables['flash_count'][:]

#declare a dim*dim numpy array
flash_count_array = np.zeros((dim,dim))

for i in range(flash_count):
    lat = file2read.variables['flash_lat'][i]
    lon = file2read.variables['flash_lon'][i]

    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
        lat_index = int((lat - lat_min) / (lat_max - lat_min) * dim)
        lon_index = int((lon - lon_min) / (lon_max - lon_min) * dim)
        flash_count_array[lat_index][lon_index] += 1


print(flash_count_array)
    