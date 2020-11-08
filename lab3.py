import data_generation as dg
import classification as cl


def classifiers_marathon(mu, sigma, sizes=None, data_name='data',
                         instances_count=1000, train_part=0.7,
                         target_class=1):
    """ Generates data samples based on the set parameters,
    draws graphs of data distribution, tests classifiers
    on the samples, and issues classification estimates.
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
