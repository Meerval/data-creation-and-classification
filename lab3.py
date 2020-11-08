import data_generation as dg
import classification as cl


def classifiers_marathon(mu, sigma, sizes=None, data_name='data',
                         instances_count=1000, train_part=0.7,
                         target_class=1):
    """ Generates data samples based on the set parameters,
    draws graphs of data distribution, tests classifiers
    on the samples, and saves classification estimates as txt.
    @param mu: the average values of creating data for every single
            feature of every single class in the specific list format like:
            [[mu_11, mu_12, ..., mu_ij, ..., mu_1n],
            [mu_21, mu_22, ..., mu_ij, ..., mu_2n],
            ...,
            [mu_m1, mu_m2, ..., mu_ij, ..., mu_mn]]
            where mu_ij is the average value of i-th class and j-th feature
    @param sigma: the standard deviations of creating data for every single
            feature of every single class in the format like average values'
            format
    @param sizes: if shape of data must be circle it should have the
            radiuses for every single feature of every single class in
            the format like average values' format
    @param data_name: str.
            Create specific names for your datasets to save
            correct saving data distribution classification results if
            work with several datasets.
    @param instances_count: int: >0
            the count of the instances for one class
    @param train_part: float: 0 <= train_part <= 1.
            The ratio of train instances count to all
            instances count.
    @param target_class: int: 0 <= target_class < count of the classes.
            The class for identification have to be less than count of
            the classes.
    """
    # Generation of Classes and labels
    samples = list(dg.get_dataset(mu, sigma, sizes, data_name,
                                  instances_count, train_part))
    # Classifications
    # Logistic Regression
    cl.get_classification('logreg', *samples, data_name, target_class)
    # Decision Tree
    cl.get_classification('tree', *samples, data_name, target_class)
    # Random Forest
    cl.get_classification('forest', *samples, data_name, target_class)


if __name__ == '__main__':
    # Normal Dataset Learning
    mu_n = [[4, 2, -4], [2, 3, -2]]
    sigma_n = [[1, 1.3, 0.9], [1, .7, 1]]
    classifiers_marathon(mu_n, sigma_n, data_name='normal')

    # Circle Dataset Learning
    mu_c = [[0, 0], [0, 0], [0, 0]]
    sigma_c = [[1, 1], [1, 1], [1, 1]]
    sizes_c = [[8, 5], [10, 7], [0, 0]]
    classifiers_marathon(mu_c, sigma_c, sizes=sizes_c,
                         data_name='circle', target_class=0)
