# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:40:44 2021

@author: Jennan Wang
"""

import sdf_helper as sh
import h5py

sdf_file_name = '0004.sdf' #sdf file name, assume call it at same folder for sdf
data = sh.getdata(sdf_file_name) #get attribute string
attri_str = str(sh.list_variables(data)).splitlines() #string processing...
variable_str_list = [] #string processing...
h5f = h5py.File(str(sdf_file_name+'.h5'), 'w') #open hdf5 file
h5f_data = [] #container for hdf5 data

for i in attri_str:
    variable_str_list.append((i.split(' '))[0])  #string processing...
    exec("h5f_data.append(data.%s.data)" % (variable_str_list[-1])) 
    #put sdf data to container for hdf5 data by calling it's variables
    h5f.create_dataset(variable_str_list[-1], data=h5f_data[-1]) #write to hdf5 file
    
h5f.close() #close hdf5 file






