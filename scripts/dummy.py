values = [50,1212,10,412,4,19,357,16,356,166,47,3,46,6]
labels = ["STANDING","UNKNOWN","GOLF","INTERACTION","CROUCHING","RUNNING","WALKING","BOXING","THROWING","RANGEOFMOTION","DANCING","BASKETBALL","WORKOUT","RACKETSPORT"]


import matplotlib.pyplot as plt
import numpy as np

import random
l = values
L = [ (l[i],i) for i in range(len(l)) ]
L.sort(reverse=True)
sorted_l,permutation = zip(*L)
new_labels = [labels[idx] for idx in permutation]

x = np.arange(len(new_labels))  # the label locations
width = 0.9  # the width of the bars

fig, ax = plt.subplots(figsize=(20,10))

rects1 = ax.bar(x, sorted_l, width, label='count')

ax.set_ylabel('minutes')
ax.set_title('minutes for each motion type')
ax.set_xticks(x)
ax.set_xticklabels(new_labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + width/2, height),
                    xytext=(2, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)

fig.tight_layout()

plt.show()
