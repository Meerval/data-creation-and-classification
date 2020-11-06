import data_generation as dg
import classification as cl


def classifiers_marathon(mu, sigma, sizes=None, data_type='normal',
                         instances_count=1000, train_part=0.7,
                         target_class=1):
    """ Generates data samples based on the set parameters,
    draws graphs of data distribution, tests classifiers
    on the samples, and issues classification estimates.
    """
    # Generation of Classes and labels
    samples = list(dg.get_dataset(mu, sigma, sizes, data_type,
                                  instances_count, train_part))
    # Classifications
    # Logistic Regression
    cl.get_classification('logreg', *samples, data_type, target_class)
    # Decision Tree
    cl.get_classification('tree', *samples, data_type, target_class)
    # Random Forest
    cl.get_classification('forest', *samples, data_type, target_class)


# Normal Dataset Learning
normal_mu = [[4, 2, -4], [2, 3, -2]]
normal_sigma = [[1, 1.3, 0.9], [1, .7, 1]]
classifiers_marathon(normal_mu, normal_sigma)

# Bad_Normal Dataset Learning
bad_normal_mu = [[3, 2, -2], [2.7, 3, -1.7], [2.2, 4, -2.3]]
bad_normal_sigma = [[2, 1.5, 1.5], [1.8, 1.5, 1.2], [2, 1.5, 1.7]]
classifiers_marathon(bad_normal_mu, bad_normal_sigma,
                     data_type='bad_normal', target_class=2)

# Circle Dataset Learning
circle_mu = [[0, 0], [0, 0], [0, 0]]
circle_sigma = [[1, 1], [1, 1], [1, 1]]
circle_sizes = [[10, 7], [8, 5], [0, 0]]
classifiers_marathon(circle_mu, circle_sigma, sizes=circle_sizes,
                     data_type='circle', target_class=1)
