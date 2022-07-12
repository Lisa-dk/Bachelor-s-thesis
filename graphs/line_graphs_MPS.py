import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import os
import numpy as np

def get_MPS_results(file):
    cb_sizes = []
    mae = []
    mae_sd = []
    cs_25 = []
    cs_25_sd = []
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
            cs_25.append(round(float(line[3]),3))
            cs_25_sd.append(round(float(line[4]),3))
            cs_0.append(round(float(line[5]),3))
            cs_0_sd.append(round(float(line[6]),3))
    return cb_sizes, mae, mae_sd, cs_25, cs_25_sd, cs_0, cs_0_sd


file = './MPS/validation_junclets.csv'
cb_sizes, mae, mae_sd, cs_25, cs_25_sd, cs_0, cs_0_sd = get_MPS_results(file)

ax = plt.subplot()
ax.plot(cb_sizes, mae, color='#1a9988')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.errorbar(cb_sizes, mae, yerr = mae_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)
ax.set_ylim(0, 15) 
ax.grid(axis = 'y')

ax.set_ylabel('MAE in years')
ax.set_xlabel('Sub-codebook size')
plt.title('Mean Absolute Error (MAE) over sub-codebook size')
plt.xticks(np.arange(0, 930, 100))
plt.yticks(np.arange(0, 15, 1)) 
plt.show()


ax = plt.subplot()
ax.plot(cb_sizes, cs_25, color='#1a9988', label='CS (α=25)')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.errorbar(cb_sizes, cs_25, yerr = cs_25_sd,fmt='o',ecolor = 'black',color='#1a9988', capsize=5)
ax.plot(cb_sizes, cs_0, color='#105f55', label='CS (α=0)')
ax.errorbar(cb_sizes, cs_0, yerr = cs_0_sd,fmt='o',ecolor = 'black',color='#105f55', capsize=5)
ax.grid(axis = 'y')
ax.set_ylim(0, 100)
ax.set_ylabel('CS (%)')
ax.set_xlabel('Sub-codebook size')
plt.title('Cumulative Score (CS) over sub-codebook size')
plt.xticks(np.arange(0, 930, 100))
plt.yticks(np.arange(0, 110, 10))
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 11})
plt.tight_layout()
plt.show()
