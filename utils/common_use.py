"""
Reload module
"""

import imp
imp.reload(module)



"""
My Favorite Boxplot
"""

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


"""
My favorite HCA
"""

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import numpy as np
#X = five_matrix.mean(axis=0).mean(axis=0)
#X = all_auroc.T
X = buffer
#Z = linkage(X, 'single', metric = 'correlation')
Z = linkage(X, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
hierarchy.set_link_color_palette(['#5e4fa2', '#66c2a5', '#f46d43', '#9e0142'])

c, coph_dists = cophenet(Z, pdist(X))
f = plt.figure(figsize=(12, 1))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
dend = hierarchy.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,
    above_threshold_color='#999999',
    color_threshold = 35
    # font size for the x axis labels
)
#plt.yticks(np.arange(0, 2, 1), np.arange(0, 2, 1), fontsize=15, rotation = 0)
#plt.xticks(np.arange(1, 121, 10), new_ticks, fontsize=15, rotation = 45)
hierarchy.set_link_color_palette(None)
f.savefig('./Figure/Fig002.pdf')
