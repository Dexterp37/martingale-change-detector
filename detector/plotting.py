import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from collections import OrderedDict

def plot_heatmap(ax, data, labels_x, labels_y, title):
    data = np.array(data).T

    ax.pcolor(data, cmap=matplotlib.cm.Blues)
    ax.set_xlim([0, data.shape[1]])
    ax.set_ylim([0, data.shape[0]])

    ax.set_xlabel('date')
    ax.set_xticklabels(labels_x if len(labels_x) == data.shape[1] else [None] + labels_x)
    ax.set_xticks(np.arange(data.shape[1]), minor=False)

    # Use the bucket label for the labels on Y.
    ax.set_yticklabels(labels_y, va='center')
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    mod = np.ceil(len(labels_y) / 20.0)
    ax.set_yticklabels([label if idx % mod == 0 else "" for idx, label in enumerate(labels_y)], va='center')

    ax.set_title(title)


def plot_data_changes_heatmap(data, changes, labels_x, labels_y, title='Changes', ax=None):
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title, fontsize=20)
    else:
        plt.sca(ax)

    # Plot the raw data.
    plot_heatmap(plt.gca(), data, labels_x, labels_y, title if ax is not None else '')
    plt.xticks(rotation=45)

    # If we're upsampling our date, space out the labels.
    num_xlabels = len(labels_x)
    num_samples = len(data)
    if num_xlabels != num_samples:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(num_samples / num_xlabels))

    # Plot the detected changes or a text if no changes were detected.
    if len(changes) > 0:
        for x in changes:
            # The 0.5 offset is needed to align with the heatmap labels.
            plt.axvline(x=x + 0.5, linestyle='dashed', color='r', label='Change event')

    # Remove the duplicated labels.
    if len(changes) > 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

    if ax is None:
        plt.show()


def plot_changes(histograms, histogram_name, changes, martingale, pvalues, strangeness,
                 augmented_data, augmentation_factor=9):
    hist_data = histograms[histogram_name]

    # Get the labels to use for the plots.
    x_labels = [h['date'][5:10] for h in hist_data]
    y_labels = [str(b) for b in hist_data[0]['buckets']]

    fig = yield [x_labels[int(np.floor(c / augmentation_factor))] for c in changes]
    if fig:
        ax = fig.add_subplot(4, 1, 1)
        plot_data_changes_heatmap(augmented_data, changes, x_labels, y_labels, histogram_name, ax)

        ax = fig.add_subplot(4, 1, 2)
        ax.plot(range(len(martingale)), martingale)
        ax.set_title('martingale')
        ax.set_xlabel('sample')
        ax.set_ylabel('value')
        ax.set_xlim([0, len(martingale)])
        ax.set_ylim(top=np.max(martingale) + 0.01)

        ax = fig.add_subplot(4, 1, 3)
        ax.plot(range(len(pvalues)), pvalues)
        ax.set_title('p-values')
        ax.set_xlabel('sample')
        ax.set_ylabel('p-value')
        ax.set_xlim([0, len(pvalues)])
        ax.set_ylim(top=np.max(pvalues) + 0.01)

        ax = fig.add_subplot(4, 1, 4)
        ax.plot(range(len(strangeness)), strangeness)
        ax.set_title('strangeness')
        ax.set_xlabel('sample')
        ax.set_ylabel('strangeness')
        ax.set_xlim([0, len(strangeness)])
        ax.set_ylim(top=np.max(strangeness) + 0.01)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, left=0.2)

        yield
