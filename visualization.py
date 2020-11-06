import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import numpy as np
import os
# Removing warning that more than 20 figures have been opened
mpl.rc('figure', max_open_warning=0)


class DataDistribution:
    """Creates scatterograms and histograms for every double
    combination of features.
    """

    def __init__(self, classes, data_type='normal',
                 save=True, show=False, xfig=8, dpi=150):
        self.classes = classes
        self.data_type = data_type
        self.save, self.show = save, show
        # The information about input data parameters
        self.class_count = len(self.classes)
        self.features_count = len(self.classes[0][0])
        self.variations = list(combinations(range(0, self.features_count), 2))
        # The information about figure dimensions
        self.xfig, self.yfig, self.dpi = xfig, None, dpi
        self.side, self.space = None, None
        self.xscatter, self.yscatter = None, None
        self.xhist, self.yhist = None, None
        self.xstart, self.xstop, self.dx = None, None, None
        self.ystart, self.ystop, self.dy = None, None, None
        # The objects of figure
        self.gs = None
        self.scatter, self.hist0, self.hist1 = None, None, None

    def draw(self):
        """ Draws, show and save scatterograms and histograms for every double
        combination of features.
        """
        for variation in self.variations:
            name = self.data_type + '_data_distribution_for_' + \
                   str(variation[0] + 1) + '&' + str(variation[1] + 1) + \
                   '_features'
            self.get_dimensions(variation)
            self.get_figure(name)
            # Creating distribution for every feature in combination
            labels = []
            for class_index in range(self.class_count):
                self.build_scatter(variation, class_index)
                self.build_hist(variation, class_index)
                self.build_hist(variation, class_index, True)
                labels.append('Class ' + str(class_index))
            plt.legend(labels=labels, bbox_to_anchor=(0., 1.),
                       loc='lower left')
            if self.save:
                get_saving(self.data_type, name, self.dpi)
            if self.show:
                plt.show()

    def get_dimensions(self, variation):
        """ Get dimensions of the figure. """
        self.side, self.space = 0.08*self.xfig, 0.02*self.xfig
        self.xscatter = 0.75*(self.xfig - 2*self.side - self.space)
        self.xhist = 0.25*(self.xfig - 2*self.side - self.space)
        self.xstart, self.xstop, self.dx = self.get_borders(variation[0])
        self.ystart, self.ystop, self.dy = self.get_borders(variation[1])
        self.yfig = (self.xscatter*self.dy / self.dx +
                     self.xhist + 2*self.side + self.space)
        self.yscatter = self.yfig - self.xhist - 2 * self.side - self.space
        self.yhist = self.xhist

    def get_borders(self, feature_index):
        """ Returns start and stop of the parameters of feature
        and its absolute difference.
        """
        start = min(self.classes[0][:, feature_index])
        stop = max(self.classes[0][:, feature_index])
        for class_index in range(1, len(self.classes)):
            instances = self.classes[class_index][:, feature_index]
            minimal, maximal = min(instances), max(instances)
            start = minimal if minimal < start else start
            stop = maximal if maximal > stop else stop
        return start, stop, abs(stop - start)

    def get_figure(self, name):
        """ Creates space for drawing scatter and histograms """
        fig = plt.figure(name, figsize=(self.xfig, self.yfig))
        fig.suptitle(name.title().replace('_', ' '), fontsize=16)
        wr, hr = (self.xscatter, self.xhist), (self.yhist, self.yscatter)
        l, r = self.side/self.xfig, 1 - self.side/self.xfig
        b, t = self.side/self.yfig, 1 - self.side/self.yfig
        ws, hs = self.space/self.xfig, self.space/self.yfig
        self.gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                                   left=l, right=r, bottom=b, top=t,
                                   wspace=ws, hspace=hs)
        self.scatter = fig.add_subplot(self.gs[1, 0])
        self.hist0 = fig.add_subplot(self.gs[0, 0], sharex=self.scatter)
        self.hist1 = fig.add_subplot(self.gs[1, 1], sharey=self.scatter)

    def build_scatter(self, features_indexes, class_index):
        """ Builds the scatter of data distribution for the features.
        """
        self.scatter.scatter(self.classes[class_index][:, features_indexes[0]],
                             self.classes[class_index][:, features_indexes[1]],
                             marker=".", alpha=0.7)
        xticks = np.arange(round(self.xstart), round(self.xstop + 2), 2)
        yticks = np.arange(round(self.ystart), round(self.ystop + 2), 2)
        self.scatter.set(xlabel='Feature ' + str(features_indexes[0] + 1),
                         ylabel='Feature ' + str(features_indexes[1] + 1),
                         xticks=xticks, yticks=yticks)
        self.scatter.grid('both', linewidth=0.5, linestyle='--')

    def build_hist(self, features_indexes, class_index, horizontal=False):
        """  Builds the histogram of data distribution for the features.
        """
        if horizontal:
            self.hist1.hist(self.classes[class_index][:, features_indexes[1]],
                            bins=self.get_bins(horizontal), alpha=0.7,
                            orientation='horizontal')
            self.hist1.set_xlabel('Count')
            self.hist1.tick_params(axis="y", labelleft=False)

        else:
            self.hist0.hist(self.classes[class_index][:, features_indexes[0]],
                            bins=self.get_bins(), alpha=0.7)
            self.hist0.set_ylabel('Count')
            self.hist0.tick_params(axis="x", labelbottom=False)

    def get_bins(self, horizontal=False, bins_count=35):
        """ Creates coordinates for histogram bins"""
        if horizontal:
            step = self.dy / (bins_count * self.dy / self.dx)
            start, stop = self.ystart, self.ystop
        else:
            step = self.dx / bins_count
            start, stop = self.xstart, self.xstop
        return np.arange(round(start), round(stop + step), step)


def probabilities_hist(probabilities_train, probabilities_test, labels_train,
                       labels_test, classifier_type='classifier',
                       data_type='data', target_class=1,
                       save=True, show=False, dpi=150):
    """ Creates histograms of the predictions' distribution. """
    fig = plt.figure(data_type + ': histograms of the' +
                     classifier_type + 'predictions', (8, 6))
    fig.suptitle('Probabilities:  ' + classifier_type.upper() +
                 ' Classification of ' +
                 data_type.title().replace('_', ' ') + ' Data',
                 fontsize=16)
    plt.subplot(211)    # Train Data Subplot
    plt.title('TRAIN Predictions')
    single_hist(probabilities_train, labels_train, target_class)
    plt.subplot(212)    # Test Data Subplot
    plt.title('TEST Predictions')
    single_hist(probabilities_test, labels_test, target_class)
    for ax in fig.get_axes():   # Remove axes between subplots
        ax.label_outer()
    if save:
        get_saving(data_type, data_type + '_' + classifier_type +
                   '_predictions_hist', dpi=dpi)
    if show:
        plt.show()


def single_hist(probabilities, labels, target_class=1):
    """ Creates histogram of the predictions' distribution for separate
    sample.
    """
    class_count = len(probabilities[target_class])
    bins = np.arange(0., 1.02, 0.02)  # Places of histograms' bins
    for predicted_class in range(class_count):
        plt.hist(probabilities[labels == predicted_class, target_class],
                 bins=bins, alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Count of Instances')


def roc(labels, probabilities, classifier_type='classifier',
        data_type='data', save=True, show=False, dpi=150):
    """ Builds ROC curve. """
    title = ('ROC Curves: ' + classifier_type.upper() +
             ' Classification of ' +
             data_type.title().replace('_', ' ') + ' Data')
    skplt.metrics.plot_roc(labels, probabilities, figsize=(6, 6),
                           title=title, title_fontsize=16)

    if save:
        get_saving(data_type, data_type + '_' + classifier_type +
                   '_ROC_curve', dpi=dpi)
    if show:
        plt.show()


def get_saving(dir_name='directory', fig_name='figure', dpi=150):
    """ Save plt into special folder."""
    directory = dir_name + '/'
    if not os.path.isdir(directory):
        os.makedirs(directory)    # Creating folder if it is not exist
    plt.savefig(directory + fig_name + '.png', dpi=dpi)
