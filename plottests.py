from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

def create_plot(file_name, axes, index, y_lim=None):
    ax = axes[index]
    file = open(file_name, 'r')

    models = []
    loss_Vals = []

    for line in file:
        strings = line.split(' ')
        model = strings[0].split('/')[-1].split('_')
        model = model[0] + '_' +model[-1]
        loss_val = strings[-1]
        loss_val_val = float(loss_val)
        models.append(model.split('_')[0])
        loss_Vals.append(loss_val_val)

    file.close()
    # fig, ax = plt.subplots(figsize=(50,20))
    # ax.yaxis.set_major_formatter(formatter)
    # ax.xticks(fontsize=7)
    title = file_name.split('/')[-1].split('.')[0]
    labels = ['GLOW', 'FF', 'VAE', 'RNN', 'RNN2']
    types = ['Basketball', 'Boxing', 'Walking', 'Throwing', 'Interaction']
    x = np.arange(len(labels))
    width = 0.18
    ax.set_xticklabels(labels)
    ax.set_xticks(x)

    for i in range(5):
        offset = 2 - i
        bar_vals = loss_Vals[i::5]
        print(len(bar_vals))
        print("")
        print(len(x))
        ax.bar(x - width * offset, bar_vals, width, label=types[i])

    ax.legend()
    if not y_lim is None:
        ax.ylim(y_lim[1], y_lim[1])
    ax.grid()
    ax.set(ylabel='mean error',
           title=title)


fig, ax = plt.subplots(10, figsize=(25,100))

folder = "test_out/rotated&shuffled"

create_plot(folder + '/mse_order_0_lower.txt', ax, 0)
create_plot(folder + '/mse_order_1_lower.txt', ax, 1)
create_plot(folder + '/mse_order_2_lower.txt', ax, 2)
create_plot(folder + '/mse_order_3_lower.txt', ax, 3)
create_plot(folder + '/mse_order_0_upper.txt', ax, 4)
create_plot(folder + '/mse_order_1_upper.txt', ax, 5)
create_plot(folder + '/mse_order_2_upper.txt', ax, 6)
create_plot(folder + '/mse_order_3_upper.txt', ax, 7)
create_plot(folder + '/feet_distances.txt',    ax, 8)
create_plot(folder + '/perceptual_loss.txt',   ax, 9, y_lim=[0,0.005])

fig.savefig(folder + "/overview.pdf")


