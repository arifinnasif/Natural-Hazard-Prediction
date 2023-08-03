import netCDF4
import os
import datetime
import numpy as np

cwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

lat_min = 33
lat_max = 37

lon_min = -89
lon_max = -85

# lat_min = -35
# lat_max = 55

# lon_min = -125
# lon_max = -58

lat_dim =int( (lat_max - lat_min) / 4 )
lon_dim =int( (lon_max - lon_min) / 4 )

dim = 50
hours_to_read = 1000
files_per_hour = 1
N_ncfiles = hours_to_read * files_per_hour



# get all files from 'glm directory' that ends with '.nc'
glm_dir = cwd+'\\glm\\'

nc_files = [f for f in os.listdir(glm_dir) if f.endswith('.nc')]
nc_files = nc_files[:N_ncfiles] # for 3 hours

"""
# write the start and end time of each file in a text file

date_output_file = open(cwd+'output.txt', 'w')
for filename in nc_files:
    file2read = netCDF4.Dataset(glm_dir+filename, 'r')

    dt_start = datetime.datetime.fromtimestamp(float(file2read.variables['product_time_bounds'][0]))
    dt_end = datetime.datetime.fromtimestamp(float(file2read.variables['product_time_bounds'][1]))

    
    date_output_file.write(str(dt_start)+','+str(dt_end)+'\n')
date_output_file.close()

"""



# to check which area is lightning prone

# flash_checker_array = np.zeros((lat_dim+1, lon_dim+1))
# for hour in range(hours_to_read):
#     for file in range(files_per_hour):
#         file2read = netCDF4.Dataset(glm_dir+nc_files[hour*files_per_hour+file], 'r')

#         flash_count = file2read.variables['flash_count'][:]

#         for i in range(flash_count):
#             lat = file2read.variables['flash_lat'][i]
#             lon = file2read.variables['flash_lon'][i]
            
#             if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
#                 lat_index = int((lat - lat_min) / 4)
#                 lon_index = int((lon - lon_min) / 4)
#                 flash_checker_array[lat_index][lon_index] += 1

# #find maximum element in flash_checker_array with its index
# max_element = np.amax(flash_checker_array)
# max_element_index = np.where(flash_checker_array == np.amax(flash_checker_array))

# print(max_element)
# print(max_element_index)
# # find lat and lon using max_element_index
# lat = lat_min + max_element_index[0][0] * 4
# lon = lon_min + max_element_index[1][0] * 4
# print(lat, lon)
# np.save(cwd+'\\flash_checker_array.npy', flash_checker_array)





# load flash_checker_array
# flash_checker_array = np.load(cwd+'\\flash_checker_array.npy')
# # find the range of lat and lon with maxm flash count
# max_element_index = np.where(flash_checker_array == np.amax(flash_checker_array))
# lat = lat_min + max_element_index[0][0] * 4
# lon = lon_min + max_element_index[1][0] * 4

# lat_f = lat_min + (max_element_index[0][0]+1) * 4
# lon_f = lon_min + (max_element_index[1][0]+1) * 4
# print(lat, lat_f)
# print(lon, lon_f)



total_flash_count = 0
# declare a numpy array of hours_to_read*1*dim*dim
flash_count_array = np.zeros((hours_to_read, 1, dim, dim))

for hour in range(hours_to_read):
    for file in range(files_per_hour):
        file2read = netCDF4.Dataset(glm_dir+nc_files[hour*files_per_hour+file], 'r')

        flash_count = file2read.variables['flash_count'][:]

        for i in range(flash_count):
            lat = file2read.variables['flash_lat'][i]
            lon = file2read.variables['flash_lon'][i]

            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                total_flash_count += 1
                lat_index = int((lat - lat_min) / (lat_max - lat_min) * dim)
                lon_index = int((lon - lon_min) / (lon_max - lon_min) * dim)
                flash_count_array[hour][0][lat_index][lon_index] += 1


np.save(cwd+'\\flash_count_array.npy', flash_count_array)
print(total_flash_count)
    