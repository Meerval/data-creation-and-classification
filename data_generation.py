import numpy as np
import visualization as vis


class DataGenerator:
    """ Generates data samples based on the set parameters. """
    def __init__(self, mu, sigma, sizes=None, data_name='data',
                 instances_count=1000, train_part=0.7):
        self.mu, self.sigma, self.sizes = mu, sigma, sizes
        self.data_name = data_name
        self.instance_count = instances_count
        self.train_part = train_part
        self.class_count = len(self.mu)
        self.features_count = len(self.mu[0])
        self.features, self.classes = None, None
        self.labels = None

    def create_features(self, i=0):
        """ Generates special features of the class index == i. """
        for j in range(self.features_count):  # j is the index of the feature
            if self.sizes:  # For circle distributed data
                angles = np.linspace(0, 2 * np.pi, num=self.instance_count)
                angles = angles.reshape(self.instance_count, 1)
                noises = np.random.normal(self.mu[i][j], self.sigma[i][j],
                                          [self.instance_count, 1])
                if j & 1:  # If j is odd
                    features = self.sizes[i][j] * np.cos(angles) + noises
                else:      # If j is even
                    features = self.sizes[i][j] * np.sin(angles) + noises
            else:   # For normally distributed data
                features = np.random.normal(self.mu[i][j], self.sigma[i][j],
                                            [self.instance_count, 1])
            if j > 0:
                features = np.hstack((self.features, features))
            self.features = features
        return self.features

    def create_classes(self):
        """ Generates classes from features."""
        for i in range(self.class_count):  # i is the index of the class
            features = self.create_features(i=i)
            if i > 0:
                features = np.vstack((self.classes, features))
            self.classes = features
        return self.classes

    def get_separated_classes(self):
        """ Divides classes into separated classes. """
        return np.split(self.classes, self.class_count)

    def create_labels(self):
        """ Creates the labels for the classes. """
        for i in range(self.class_count):  # i is the index of the class
            labels = np.array([i] * self.instance_count)
            if i > 0:
                labels = np.hstack((self.labels, labels)).ravel()
            self.labels = labels
        return self.labels

    def samples_split(self):
        """ Splits classes for training and test samples and returns
        features and labels for them.
        """
        instance_count = len(self.labels)
        # Creating indexes for each instance
        indexes = np.arange(instance_count)
        np.random.default_rng().shuffle(indexes)
        self.classes = self.classes[indexes]
        self.labels = self.labels[indexes]
        train_count = int(self.train_part * instance_count)
        features_train = self.classes[0: train_count]
        features_test = self.classes[train_count:]
        labels_train = self.labels[0: train_count]
        labels_test = self.labels[train_count:]
        return features_train, features_test, labels_train, labels_test

    def draw_graphics(self):
        """ Calls the function for creating scatterograms and histograms
        of data distribution.
        """
        vis.DataDistribution(self.get_separated_classes(), self.data_name).draw()


def get_dataset(mu, sigma, sizes=None, data_name='data',
                instance_count=1000, train_part=0.7):
    """ Instantiates the DataGenerator class and returns features and
    labels for train and test samples.
    """
    data = DataGenerator(mu, sigma, sizes, data_name,
                         instance_count, train_part)
    data.create_classes()
    data.create_labels()
    data.draw_graphics()
    return data.samples_split()
