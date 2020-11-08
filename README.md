# data-creation-and-classification
This project has created as my tasks for the university laboratory works. Welcome! It is my starting point for data analysis and starting using GitHub))
## Project objective
This repository about creation of datasets and its classification
. Classification methods such as *__logistic regression__*, *__decision tree__* and *__random forest__* are used here.
With this code you can create two types of data distribution (*__centred__* and *__circle__*) with using different parameters for the final shape.  
<img src="https://github.com/Meerval/data-creation-and-classification/blob/master/normal/normal_data_distribution_for_1%262_features.png" alt="drawing" width="500"/> <img src="https://github.com/Meerval/data-creation-and-classification/blob/master/circle/circle_data_distribution_for_1%262_features.png" alt="drawing" width="500"/>  
After creating data you will get the similar images for every way of classification:  
<img src="https://github.com/Meerval/data-creation-and-classification/blob/master/normal/normal_forest_predictions_hist.png" alt="drawing" width="500"/> <img src="https://github.com/Meerval/data-creation-and-classification/blob/master/normal/normal_forest_ROC_curve.png" alt="drawing" width="400"/>  
and table with estimations of the classifiers as .txt file like this but don't so beautiful:

	Normal Dataset: FOREST Classifier Estimations            

|                    |Accuracy|Sensitivity|Specificity|
|--------------------|-------:|----------:|----------:|
|Train (1400 samples)|    1.00|       1.00|       1.00|
|Test (600 samples)  |    0.94|       0.92|       0.95|

 	AUC forest: 0.98; Trees' count: 300
  
## Using manual
To start a project use file __lab3.py__. You can find there the start code like this:
```python
if __name__ == '__main__':
    # Normal Dataset Learning
    mu = [[4, 2, -4], [2, 3, -2]]
    sigma = [[1, 1.3, 0.9], [1, .7, 1]]
    classifiers_marathon(mu, sigma, data_name='normal')
```
the function *__ classifiers_marathon()__* was described in the its
 documentation.
## Warning!
If you run lab3.py on your computer, it will create the path in your
 repository with images and tables as above for every name of data (if you choose data_name='normal',
a folder named 'normal' will be created).
