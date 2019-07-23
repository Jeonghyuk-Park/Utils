# Reload module

import imp
imp.reload(module)



# My Favorite Boxplot

import matplotlib.pyplot as plt
import numpy as np
boxprops = dict(linestyle='-', linewidth=2.5, color='black')
flierprops = dict(marker='+', markerfacecolor='black', markersize=3,
                  linestyle='none')
medianprops = dict(linestyle='-', linewidth=2.5, color='black')
capprops = dict(linestyle='-', linewidth=2.5, color='black')
whiskerprops = dict(linestyle='-', linewidth=2.5, color='black')
widths = .5
# Random test data
all_data1 = [data1, data2, data3]
all_data2 = [data4, data5, data6]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

# rectangular box plot
bplot1 = axes[0].boxplot(all_data1,
                         vert=True,  # vertical box alignment
                         patch_artist=True,
                         boxprops = boxprops,
                         capprops = capprops,
                         flierprops = flierprops,
                         whiskerprops = whiskerprops,
                         medianprops = medianprops,# fill with color
                         widths=widths)  # will be used to label x-ticks

bplot2 = axes[1].boxplot(all_data2,
                         vert=True,  # vertical box alignment
                         patch_artist=True,
                         boxprops = boxprops,
                         capprops = capprops,
                         flierprops = flierprops,
                         whiskerprops = whiskerprops,
                         medianprops = medianprops,# fill with color
                         widths=widths)  # will be used to label x-ticks

# fill with colors
colors = ['#5e4fa2', '#66c2a5', '#f46d43']
for bplot in [bplot1,bplot2]:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
fig.savefig('./Figure/GCLtrain_07.svg', format='svg')
