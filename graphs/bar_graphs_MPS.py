
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
plt.rcParams["font.size"] = "12"
import os
import numpy as np

def get_MPS_results(file):
    features = []
    mae = []
    cs_25 = []
    cs_0 = []

    with open(file) as f:
        for line in f.readlines():
            line = line.split(',')
            if line[0] == '':
                continue
            features.append(line[0])
            mae.append(round(float(line[1]), 1))
            cs_25.append(round(float(line[2]),1))
            cs_0.append(round(float(line[3]),1))
    return features, mae, cs_25, cs_0
    

file = './new/MPS/test.csv'
features, mae, cs_25, cs_0 = get_MPS_results(file)
file = './new/MPS/test_aug.csv'
features, mae_aug, cs_25_aug, cs_0_aug = get_MPS_results(file)

x = np.arange(len(features))  # the label locations
width = 0.3  # the width of the bars

ax = plt.subplot()

rects1 = ax.barh(x - width/1.7, width=mae_aug, height=width, label='Augmented', color='#105f55')
rects2 = ax.barh(x + width/2, width=mae, height=width, label='Non-augmented', color='#1a9988')

ax.set_ylabel('Features')
ax.set_xlabel('MAE in years')
plt.title('Mean Absolute Error (MAE) per feature')
ax.set_yticks(x, features)
ax.set_xlim(0,25)
ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})


ax.bar_label(rects1, padding=3, fontsize=10)
ax.bar_label(rects2, padding=3, fontsize=10)

plt.tight_layout()
plt.show()


x = np.arange(len(features)) 
width = 0.3 

ax = plt.subplot()

rects1 = ax.barh(x - width/1.7, width=cs_25_aug, height=width, label='Augmented', color='#105f55')
rects2 = ax.barh(x + width/2, width=cs_25, height=width, label='Non-augmented', color='#1a9988')
ax.set_ylabel('Features')
ax.set_xlabel('CS (%)')
plt.title('Cumulative Score (CS) (α = 25) per feature')
ax.set_yticks(x, features)
ax.set_xlim(0,105)

ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

ax.bar_label(rects1, padding=3, fontsize=10)
ax.bar_label(rects2, padding=3, fontsize=10)

plt.tight_layout()

plt.show()


x = np.arange(len(features))  
width = 0.3  

ax = plt.subplot()
rects1 = ax.barh(x - width/1.7, width=cs_0_aug, height=width, label='Augmented', color='#105f55')
rects2 = ax.barh(x + width/2, width=cs_0, height=width, label='Non-augmented', color='#1a9988')
ax.set_ylabel('Features')
ax.set_xlabel('CS (%)')
plt.title('Cumulative Score (CS) (α = 0) per feature')
ax.set_yticks(x, features)
ax.set_xlim(0,105)

ax.set_axisbelow(True)
ax.grid(zorder=3, axis='x')

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})

ax.bar_label(rects1, padding=3, fontsize=10)
ax.bar_label(rects2, padding=3, fontsize=10)

plt.tight_layout()

plt.show()
