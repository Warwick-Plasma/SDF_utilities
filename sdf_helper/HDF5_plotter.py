import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('data.h5','r')   #開啟h5檔案
print(f.keys())                            #可以檢視所有的主鍵
domain = f['Electric_Field_Ey'][:]                    #取出主鍵為data的所有的鍵值
f.close()

plt.contourf(domain)
plt.colorbar()
plt.show()
plt.clf()