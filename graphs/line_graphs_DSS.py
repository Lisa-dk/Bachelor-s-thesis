
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import os
import numpy as np

def get_DSS_results(file):
    cb_sizes = []
    mae = []
    mae_sd = []
    cs_0 = []
    cs_0_sd = []

    with open(file) as f:
        for line in f.readlines():
            line = line.split(',')
            if line[0] == 'subcodebook size':
                continue
            cb_sizes.append(int(line[0])*int(line[0]))
            mae.append(round(float(line[1]), 3))
            mae_sd.append(round(float(line[2]), 3))
            cs_0.append(round(float(line[3]),3))
            cs_0_sd.append(round(float(line[4]),3))
    return cb_sizes, mae, mae_sd, cs_0, cs_0_sd

file = './DSS_results/validation_junclets.csv'
cb_sizes, mae, mae_sd, cs_0, cs_0_sd = get_DSS_results(file)

ax = plt.subplot()
ax.plot(cb_sizes, mae, color='#1a9988')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.errorbar(cb_sizes, mae, yerr = mae_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)
ax.set_ylim(0, 75) 
ax.grid(axis = 'y')

ax.set_ylabel('MAE in years')
ax.set_xlabel('Sub-codebook size')
plt.title('Mean Absolute Error (MAE) over sub-codebook size')
plt.xticks(np.arange(0, 930, 100))
plt.yticks(np.arange(0, 75, 10)) 
plt.show()


ax = plt.subplot()
ax.plot(cb_sizes, cs_0, color='#1a9988')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.errorbar(cb_sizes, cs_0, yerr = cs_0_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)
ax.grid(axis = 'y')
ax.set_ylim(0, 100)
ax.set_ylabel('CS (%)')
ax.set_xlabel('Sub-codebook size')
plt.title('Cumulative Score (α = 0) over sub-codebook size')
plt.xticks(np.arange(0, 930, 100))
plt.yticks(np.arange(0, 110, 10))
plt.show()


