from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas
import numpy as np
import visualization as vis


class Classifier:
    """ Makes classification and get figure of probabilities distribution
    and table of classifier estimations.
    """

    def __init__(self, classifier_type, features_train, features_test,
                 labels_train, labels_test, data_name='data',
                 target_class=1):
        self.classifier_type = classifier_type
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.data_name = data_name
        self.class_count = len(np.unique(self.labels_test))
        self.target_class = target_class
        self.clf = None
        self.probabilities_train, self.probabilities_test = None, None
        self.predictions_train, self.predictions_test = None, None
        self.auc = None
        self.acc, self.sens, self.spec = None, None, None
        self.tree_depth = None if self.classifier_type != 'tree' else 20
        self.trees_count = None

    def learning_logreg(self):
        """ Makes learning of Logistic Regression Classifier. """
        clf = LogisticRegression(random_state=8, solver='saga', max_iter=1000)
        self.clf = clf.fit(self.features_train, self.labels_train)
        return self.clf

    def learning_tree(self):
        """ Makes learning of Tree Decision Classifier. """
        clf = DecisionTreeClassifier(random_state=0, max_depth=self.tree_depth)
        self.clf = clf.fit(self.features_train, self.labels_train)
        self.probabilities()
        auc_test, auc_train = self.get_auc(), self.get_auc('train')
        if 0.95 * auc_train < auc_test or self.tree_depth == 2:
            # For normal level of retraining
            return self.clf
        else:
            # Decreasing depth of the tree for high level of retraining
            self.tree_depth -= 1
            self.learning_tree()

    def learning_forest(self):
        """ Makes learning of Random Forest Classifier. """
        max_auc, max_clf = 0, None
        for trees_count in range(10, 310, 10):  # Find the best auc
            clf = RandomForestClassifier(random_state=0,
                                         n_estimators=trees_count)
            self.clf = clf.fit(self.features_train, self.labels_train)
            self.probabilities()
            auc = self.get_auc()
            if auc > max_auc:
                max_auc, max_clf = auc, self.clf
                self.trees_count = trees_count
        self.clf = max_clf
        return self.clf  # returns clf with the best auc

    def probabilities(self):
        """ Creates probabilities (from 0 to 1) for test and train. """
        self.probabilities_train = self.clf.predict_proba(self.features_train)
        self.probabilities_test = self.clf.predict_proba(self.features_test)
        return self.probabilities_train, self.probabilities_test

    def predictions(self):
        """ Creates predictions (0 or 1) for test and train. """
        self.predictions_train = self.clf.predict(self.features_train)
        self.predictions_test = self.clf.predict(self.features_test)
        return self.predictions_train, self.predictions_test

    def get_auc(self, sample_type='test'):
        """ Counts the area under ROC-curve. """
        if sample_type == 'test':
            labels, probabilities = self.labels_test, self.probabilities_test
        else:
            labels, probabilities = self.labels_train, self.probabilities_train
        # Calculating the area under the curve for 2 classes
        if self.class_count == 2:
            self.auc = roc_auc_score(labels,
                                     probabilities[:, self.target_class])
        # Calculating the area under the curve for several classes (> 2)
        else:
            self.auc = roc_auc_score(labels, probabilities, multi_class='ovo')
        return self.auc

    def classifier_estimation(self, sample_type='test'):
        """Creates estimates such as accuracy, sensitivity and
        specificity and returns them.
        """
        self.predictions()
        if sample_type == 'test':
            labels, predictions = self.labels_test, self.predictions_test
        else:
            labels, predictions = self.labels_train, self.predictions_train
        # Makes flags (True or False) for every singe label and prediction
        labels = np.array([True if i == self.target_class else False
                           for i in labels])
        predictions = np.array([True if i == self.target_class else False
                                for i in predictions])
        tp = sum(labels & predictions)  # 1 & 1
        tn = sum(~labels & ~predictions)  # 0 & 0
        fp = sum(~labels & predictions)  # 0 & 1
        fn = sum(labels & ~predictions)  # 1 & 0
        self.acc = round((tp + tn) / (tp + tn + fp + fn), 2)
        self.sens = round(tp / (tp + fn), 2)
        self.spec = round(tn / (tn + fp), 2)
        return self.acc, self.sens, self.spec

    def draw_graphics(self):
        """ Calls the functions for creating histograms of predictions
        distribution and drawing ROC-curve.
        """
        self.probabilities()
        vis.probabilities_hist(self.probabilities_train, self.probabilities_test,
                               self.labels_train, self.labels_test,
                               self.classifier_type, self.data_name,
                               self.target_class)
        vis.roc(self.labels_test, self.probabilities_test,
                self.classifier_type, self.data_name)

    def draw_table(self):
        """ Creates table for results of estimations.
        """
        estimations_train = list(self.classifier_estimation('train'))
        estimations_test = list(self.classifier_estimation('test'))
        all_estimations = np.array([estimations_train, estimations_test])
        title = '\t' + self.data_name.title().replace('_', ' ') + \
                ' Dataset: ' + self.classifier_type.upper() + \
                ' Classifier Estimations\n\n'
        # Creating table
        columns = ['Accuracy', 'Sensitivity', 'Specificity']
        indexes = ['Train (' + str(len(self.labels_train)) + ' samples)',
                   'Test (' + str(len(self.labels_test)) + ' samples)']
        table = pandas.DataFrame(all_estimations, indexes, columns)
        # Save table as txt
        filename = (self.data_name + '/' +
                    self.data_name + '_classifier_estimate.txt')
        message = (title + str(table) + '\n\tAUC ' +
                   self.classifier_type + ': ' + str(round(self.get_auc(), 2)))
        try:  # Getting count of lines in the file if it is exist
            with open(filename, 'r') as file:
                text = file.readlines()
                lines_count = len(text)
        except FileNotFoundError:
            lines_count = 0
        # Saving of the table. If lines_count < 23 overwrites the file
        # to save only the latest results after restarting the program
        with open(filename, 'a' if lines_count < 23 else 'w') as file:
            if self.classifier_type == 'tree':
                message += (';\tTree Depth: ' + str(self.tree_depth))
            elif self.classifier_type == 'forest':
                message += (";\tTrees' count: " + str(self.trees_count))
            message += '\n\n\n'
            file.write(message)


def get_classification(classifier_type, features_train, features_test,
                       labels_train, labels_test, data_name='data',
                       target_class=1):
    """ Instantiates the Classifier class, saves figure of probabilities
    distribution and table of classifier results estimations.
    """
    classifier = Classifier(classifier_type,
                            features_train, features_test,
                            labels_train, labels_test,
                            data_name, target_class)
    if classifier_type == 'logreg':
        classifier.learning_logreg()
    elif classifier_type == 'tree':
        classifier.learning_tree()
    else:
        classifier.learning_forest()
    classifier.draw_graphics()
    classifier.draw_table()
