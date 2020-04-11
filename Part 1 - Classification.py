# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from boruta import BorutaPy

### Read data from a file
Dataset = pd.read_csv('arrhythmia.data',header=None, na_values="?")

print(Dataset.info())


print(Dataset.dtypes)


#####feature pre-proccesing#####


## print columns missing values and percentage per column
missingCols=Dataset.columns[Dataset.isna().any()]
print('Features with missing values=',missingCols)

percent_missing = Dataset.isnull().sum() * 100 / len(Dataset)
missing_value_df = pd.DataFrame({'column_name': Dataset.columns,
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)



#replace missing values of a column with the median of that column
#do this for all columns
rows, cols = Dataset.shape
for i in range(0, cols-1):
    Dataset[i].fillna(value=Dataset[i].median(),inplace=True)



# replace all values over 2 with 1(arrhythmia problem) and rest as 1 (no problem) and limit the classes
def Arrhythmia_Problem(quality):
    if quality >= 2:
        return 1
    else:
        return 0

#Create Targets Dataset
Target_Values=pd.DataFrame(Dataset[279].apply(Arrhythmia_Problem))


#delete target col from original Dataset
del Dataset[279]

# Find the importance of each feature by using the
# boruta wrapper method

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(criterion='entropy',n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1,max_iter=120, perc=90)


# find all relevant features
feat_selector.fit(Dataset.to_numpy(),Target_Values.to_numpy().ravel())

# The number of selected features.
feat_selected=feat_selector.n_features_

# check selected features - The mask of selected features - only confirmed ones are True.
feat=feat_selector.support_

# check ranking of features
feat_rank=feat_selector.ranking_

# call transform() on X to filter it down to selected features
Dataset_filtered = feat_selector.transform(Dataset.to_numpy())


#split the data
x_train, x_test, y_train, y_test = train_test_split (Dataset_filtered ,Target_Values, test_size=0.3, random_state=1)


#standarize the data
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test= std.transform(x_test)


###############################################################################
#Define Decision Tree
clfDecisionTree =  tree.DecisionTreeClassifier(criterion='gini', max_depth=10,splitter='best', random_state=1)


#train the classifier Decision Tree                       
clfDecisionTree.fit(x_train, y_train)


#test the trained model on the test set
y_test_pred_DesicionTree=clfDecisionTree.predict(x_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred_DesicionTree))
print ('\n')
confMatrixTestDecisionTree=confusion_matrix(y_test, y_test_pred_DesicionTree, labels=None)
print ('Conf matrix Decision Tree')
print (confMatrixTestDecisionTree)
print ('\n')
# Measures of performance: Precision, Recall, F1
print ('Tree: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_DesicionTree, average='macro'))
print ('Tree: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_DesicionTree, average='micro'))
print ('\n')


###############################################################################
#Define a Naive Bayes
clfNB = GaussianNB()

clfNB.fit(x_train, y_train)

y_test_pred_NB=clfNB.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred_NB))
print ('\n')
confMatrixTestNB=confusion_matrix(y_test, y_test_pred_NB, labels=None)
print ('Conf matrix Naive Bayes')
print (confMatrixTestNB)
print ('\n')
# Measures of performance: Precision, Recall, F1
print ('Naive Bayes: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro'))
print ('Naive Bayes: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NB, average='micro'))
print ('\n')


###############################################################################
#Define Neural Network
clfNeuralNetwork = MLPClassifier(solver='lbfgs', activation='relu',
                     tol=1e-4,
                     hidden_layer_sizes=(5,5,5,5), random_state=1, max_iter=120)


clfNeuralNetwork.fit(x_train, y_train)

y_test_pred_NeuralNetwork=clfNeuralNetwork.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred_NeuralNetwork))
print ('\n')
confMatrixTestNeuralNetwork=confusion_matrix(y_test, y_test_pred_NeuralNetwork, labels=None)
print ('Conf matrix Neural Network')
print (confMatrixTestNeuralNetwork)
print ('\n')
# Measures of performance: Precision, Recall, F1
print ('Neural Network: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NeuralNetwork, average='macro'))
print ('Neural Network: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NeuralNetwork, average='micro'))
print ('\n')


###############################################################################

#Define Support vector machine

#kernel='rbf', or 'poly'
#degree: refers to the degree of the polynomial kernel
clfSVM= svm.SVC(class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, random_state=1, shrinking=True,
    verbose=False, probability=True)


clfSVM.fit(x_train, y_train)

y_test_pred_SVM=clfSVM.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred_SVM))
print ('\n')
confMatrixTestNeuralNetwork=confusion_matrix(y_test, y_test_pred_SVM, labels=None)
print ('Conf matrix Support Vector Machine')
print (confMatrixTestNeuralNetwork)
print ('\n')
# Measures of performance: Precision, Recall, F1
print ('Support Vector Machine: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_SVM, average='macro'))
print ('Support Vector Machine: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_SVM, average='micro'))
print ('\n')

###############################################################################

#random Forest
clfRF = RandomForestClassifier(criterion='gini', class_weight='balanced', max_depth=5,random_state=1)

clfRF.fit(x_train, y_train)

y_test_pred_RF=clfRF.predict(x_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred_RF))
print ('\n')
confMatrixTestNeuralNetwork=confusion_matrix(y_test, y_test_pred_RF, labels=None)
print ('Conf matrix Random Forest')
print (confMatrixTestNeuralNetwork)
print ('\n')
# Measures of performance: Precision, Recall, F1
print ('Random Forest: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF, average='macro'))
print ('Random Forest: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF, average='micro'))
print ('\n')

###############################################################################

# Prepare data for the ROC curves for Desicion Tree
#
pr_y_test_pred_DesicionTree=clfDecisionTree.predict_proba(x_test)
pr_y_test_pred_NaiveBayes=clfNB.predict_proba(x_test)
pr_y_test_pred_NeuralNetwork=clfNeuralNetwork.predict_proba(x_test)
pr_y_test_pred_SVM=clfSVM.predict_proba(x_test)
pr_y_test_pred_RF=clfRF.predict_proba(x_test)

#ROC curve: Class 1 (the minority class) is considered the "Positive"
#class in this problem
fprDesicionTree, tprDesicionTree, thresholdsDesicionTree = roc_curve(y_test, pr_y_test_pred_DesicionTree[:,1])
fprNaiveBayes, tprNaiveBayes, thresholdsNaiveBayes = roc_curve(y_test, pr_y_test_pred_NaiveBayes[:,1])
fprNeuralNetwork, tprNeuralNetwork, thresholdsNeuralNetwork = roc_curve(y_test, pr_y_test_pred_NeuralNetwork[:,1])
fprSVM, tprSVM, thresholdsSVM = roc_curve(y_test, pr_y_test_pred_SVM[:,1])
fprRF, tprRF, thresholdsRF = roc_curve(y_test, pr_y_test_pred_RF[:,1])

#line widt, lw=1
lw=2
plt.figure(4)
plt.plot(fprDesicionTree,tprDesicionTree,color='blue',label='Decision Tree')
plt.plot(fprNaiveBayes,tprNaiveBayes,color='magenta',label='Naive Bayes')
plt.plot(fprNeuralNetwork,tprNeuralNetwork,color='black',label='Neural Network')
plt.plot(fprSVM,tprSVM,color='red',label='SVM')
plt.plot(fprRF,tprRF,color='green',label='RF')



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print('Decision Tree AUC=',round(auc(fprDesicionTree,tprDesicionTree),3))
print('Naive Bayes AUC=',round(auc(fprNaiveBayes,tprNaiveBayes),3))
print('Neural Network AUC=',round(auc(fprNeuralNetwork,tprNeuralNetwork),3))
print('SVM AUC=',round(auc(fprSVM,tprSVM),3))
print('Random Forest AUC=',round(auc(fprRF,tprRF),3))





