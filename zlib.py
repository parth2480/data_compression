import zlib
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# import os

data_df = pd.read_csv('cleaned_data/C1_160308.txt', )
data_df.columns = ['Time', r'$B_x$', r'$B_y$', r'$B_z$']
df_chunk = data_df[0:int(64 * 1.5)]
data_df.astype({r'$B_y$': 'float64'})

dtype = 'float32'

# t_data = np.array(np.array(data_df['Time'].tolist(), dtype=dtype) * 1000, dtype='int16')
B_x = np.array(data_df[r'$B_x$'].tolist(), dtype=dtype)
B_y = np.array(data_df[r'$B_y$'].tolist(), dtype=dtype)
B_z = np.array(data_df[r'$B_z$'].tolist(), dtype=dtype)


"""
zlib Compression
"""

t_sum = np.zeros(10)
x_sum = np.zeros(10)
y_sum = np.zeros(10)
z_sum = np.zeros(10)
zlib_t = np.zeros(10)

for i in range(10):
    zlib_start_t = time.clock()
    # comp_t = zlib.compress(t_data, i)
    comp_x = zlib.compress(B_x, i)
    comp_y = zlib.compress(B_y, i)
    comp_z = zlib.compress(B_z, i)

    # t_sum[i] = sys.getsizeof(comp_t)
    x_sum[i] = sys.getsizeof(comp_x)
    y_sum[i] = sys.getsizeof(comp_y)
    z_sum[i] = sys.getsizeof(comp_z)
    zlib_t[i] = time.clock() - zlib_start_t

print(zlib_t)
# exit()
# print(t_sum, x_sum, y_sum, z_sum)

total_comp = x_sum + y_sum + z_sum
total_uncomp = sys.getsizeof(B_x) + sys.getsizeof(B_y) + sys.getsizeof(B_z)
kappa = total_comp / total_uncomp
print(kappa)

ind = np.arange(10)
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(ind - 2 * width, t_sum, width=width, align='edge', color='skyblue')
ax.bar(ind - width, x_sum, width=width, align='edge', color='darkgoldenrod')
ax.bar(ind, y_sum, width=width, align='edge', color='darkolivegreen')
ax.bar(ind + width, z_sum, width=width, align='edge', color='lightslategrey')
# ax.set_ylim([0, 500])
ax.set_xticks(ind)
ax.set_xlabel('Compression Level')
ax.set_ylabel('Size of data chunk (Kb)')
ax.legend([r'time$(t)$', r'$B_x$', r'$B_y$', r'$B_z$'])
plt.show()

f, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(ind, zlib_t, marker='x', s=35, color='dodgerblue')
ax1.legend(['Compression time vs level'])
ax1.set_xlabel('zlib Compression level')
ax1.set_ylabel(r'$\mathcal{t} \rightarrow$ compression time')
ax1.set_xticks(ind)
ax1.grid()
plt.show()
