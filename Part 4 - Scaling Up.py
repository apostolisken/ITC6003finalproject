# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:27:44 2020

@author: Dimitrios Tsagkarakis
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime
import dateutil
import time
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import datasets 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection  import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif



# Load data from csv
YC_clicks_100K = pd.read_csv('yoochoose-clicks.dat', 
                         low_memory=False, header=None)
YC_buys_100K = pd.read_csv('yoochoose-buys.dat', 
                       low_memory=False, header=None)

YC_clicks_100K=YC_clicks_100K.sample(1000000)


# Add column headers to features 
YC_buys_100K.columns = ['Session_ID', 'Timestamp_buys', 'Item_ID', 'Price', 'Quantity']
print(YC_buys_100K.head())  # check that headers are added

YC_clicks_100K.columns = ['Session_ID', 'Timestamp_clicks', 'Item_ID', 'Category']
print(YC_clicks_100K.head())  # check that headers are added

# Replace all values over 0 as 1(buy) and 0(no buy)
def S(i):
    if i !='S':
        if int(i)<=12:
            return i
        else:
            return str(0)
    else:
        return str(13)

YC_clicks_100K['Category']=YC_clicks_100K['Category'].apply(S)


# sumif/consolidate YC_clicks3
YC_clicks4 = YC_clicks_100K.groupby(["Session_ID", "Item_ID"])["Timestamp_clicks"].apply(list).reset_index(name='TS_CL')
YC_clicks5 = YC_clicks_100K.groupby(["Session_ID", "Item_ID"])["Category"].apply(list).reset_index(name='Cat_CL')

# outer join for clicks
YC_clicks_j = pd.merge(YC_clicks4, YC_clicks5, how = 'outer', on = ['Session_ID', 'Item_ID'])


# buys dataset add calculated features into 'pop' helper dataset
YC_buys_100K['Ttl_item_price'] = YC_buys_100K['Price']*YC_buys_100K['Quantity']

YC_pop = YC_buys_100K.groupby(['Item_ID']).sum()

YC_pop['total'] = YC_pop['Quantity'].sum()
YC_pop["frequency"] = YC_pop['Quantity']/YC_pop['total']*100
YC_pop['Avg_price_ttl'] = YC_buys_100K['Ttl_item_price'] / YC_pop['total']

YC_pop['Ttl_item_qty'] = YC_pop['Quantity']
YC_pop = YC_pop.drop(['Session_ID', 'total', 'Price', 'Quantity'], axis=1)


# consolidate YC_buys_100K dataset
YC_buys4 = YC_buys_100K.groupby(["Session_ID", "Item_ID"])["Timestamp_buys"].apply(list).reset_index(name='TS_BL')
YC_buys5 = YC_buys_100K.groupby(["Session_ID", "Item_ID"])["Price"].apply(list).reset_index(name='Price_BL')
YC_buys6 = YC_buys_100K.groupby(["Session_ID", "Item_ID"])["Quantity"].apply(list).reset_index(name='Qty_BL')

# outer join for buys dataset
YC_buys_7 = pd.merge(YC_buys4, YC_buys5, how = 'outer', on = ['Session_ID', 'Item_ID'])
YC_buys_j = pd.merge(YC_buys_7, YC_buys6, how = 'outer', on = ['Session_ID', 'Item_ID'])


# outer join for all buy/clicks/pop datasets
YC_data_init = pd.merge(YC_buys_j, YC_clicks_j, how = 'outer', on = ['Session_ID', 'Item_ID'])
YC_data = pd.merge(YC_data_init, YC_pop, how = 'outer', on = ['Item_ID'])

# check datatypes
print(YC_data.dtypes)

# Add average price feature
YC_data['Avg_price_ttl'] = YC_data['Ttl_item_price'] / YC_data['Ttl_item_qty']

# replace NaN
YC_data.fillna(0, inplace = True)

# add time, date, and duration features to df
# timestamps are sorted by default
YC_data['Number_clicks'] = YC_data['TS_CL'].str.len()
YC_data['Number_buys'] = YC_data['TS_BL'].str.len()

YC_data['First_click'] = YC_data['TS_CL'].str[0]
YC_data['Last_click'] = YC_data['TS_CL'].str[-1]

YC_data['First_click'] = pd.to_datetime(YC_data['First_click'])
YC_data['Last_click'] = pd.to_datetime(YC_data['Last_click'])
       

YC_data['Click length'] = YC_data['Last_click'] - YC_data['First_click']
YC_data['Click length'].fillna(value=YC_data['Click length'].median(),inplace=True)

YC_data['Avg_btw_clicks'] = YC_data['Click length'] / YC_data['Number_clicks']

YC_data['Month_first_click'] = pd.DatetimeIndex(YC_data['First_click']).month
YC_data['Day_first_click'] = pd.DatetimeIndex(YC_data['First_click']).day
YC_data['Hour_first_click'] = pd.DatetimeIndex(YC_data['First_click']).hour

# YC_data['Month_last_click'] = pd.DatetimeIndex(YC_data['Last_click']).month
YC_data['Day_last_click'] = pd.DatetimeIndex(YC_data['Last_click']).day
YC_data['Hour_last_click'] = pd.DatetimeIndex(YC_data['Last_click']).hour


# add global price average feature
a = YC_data['Price_BL'].to_numpy()
b = YC_data['Qty_BL'].to_numpy()
    
a = np.array([np.array(i) for i in a])
b = np.array([np.array(i) for i in b])

c = a*b
c = c.flatten()

d = [np.sum(i) for i in c]

YC_data['Ttl_purchase'] = d

    
Total_purchases = YC_data['Ttl_purchase'].sum()

e = b.flatten()
f = [np.sum(i) for i in e]

YC_data['Ttl_quantity'] = f

Total_quantity = YC_data['Ttl_quantity'].sum()

Global_avg_price = Total_purchases / Total_quantity



# Replace all values over 0 as 1(buy) and 0(no buy) - Class
def buy(i):
    if i > 0:
        return 1
    else:
        return 0

YC_data['Number_buys'].fillna(0, inplace = True)

YC_data['Class'] = YC_data['Number_buys'].apply(buy)

YC_data['Avg_price_vs_plobal_avg'] = YC_data['Avg_price_ttl'] / Global_avg_price




# category is static for combination of item ID/session ID
YC_data['Category'] = YC_data['Cat_CL'].str[0]
YC_data['Category'].fillna(0, inplace = True)


# Using One Hot Encoding for handling categorical data
# Several categories which can't be easily ordered or expressed numerically
YC_data = pd.get_dummies(YC_data, columns=['Category'], prefix=['Category'])


# drop unutilized columns
traindata = YC_data.drop(['Session_ID', 'TS_BL', 'Price_BL', 'Qty_BL', 'TS_CL',
                          'Cat_CL', 'Ttl_item_price', 'Number_buys', 'First_click', 'Last_click',
                          'Ttl_purchase', 'Ttl_quantity'], axis=1)


# truncate item id to not cause "number too high" errors - last 4 digits
traindata['Item_trunc'] = traindata['Item_ID'].astype(str)
traindata['Item_trunc'] = traindata['Item_trunc'].str[-4:]
traindata['Item_ID'] = traindata['Item_trunc']
traindata = traindata.drop(['Item_trunc'], axis=1)


# convert datetime to seconds
traindata['Click length'] = traindata['Click length'].dt.total_seconds()
traindata['Avg_btw_clicks'] = traindata['Avg_btw_clicks'].dt.total_seconds()


traindata.fillna(0, inplace = True)

traindata['Buy/Not:1/0'] = traindata['Class']
traindata = traindata.drop(['Class'], axis=1)


# balance dataset / bige number of buys - subsample clicks to get equivalent data
# count unique buys = 1 and sample equal number of buys = 0
traindata = traindata.sort_values(by ='Buy/Not:1/0', ascending=False)
Total_buys = traindata['Buy/Not:1/0'].sum()
traindata = traindata[:Total_buys*2]


# Check again that the correct columns are in the dataframe


# plot the heatmap and annotation on it
Var_Corr = traindata.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

fig = plt.gcf()  # or by other means, like plt.subplots
figsize = fig.get_size_inches()
fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5

ax = sns.heatmap(
    Var_Corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



# Naive Bayes
y = traindata['Buy/Not:1/0']
x = traindata.iloc[:,:-1]

traindata.fillna(0, inplace = True)
print(traindata.dtypes)

x1 = x.astype(float)
y1 = y.astype(int)

# Find information content of each feature
# Find the information content of each of the input variables
information = mutual_info_classif(x1, y1)
print('Information=', information)


# drop columns after mutual info - not necessary, but could remove a few elements
# traindata = traindata.drop(['xxxxxxxx'], axis=1)

# normalize / scale
# make all columns with 0-mean, and 1-std
x1 = preprocessing.scale(x1)


# split the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split (x1,y1, test_size=0.3, random_state=42)
n_samples = x_train.shape[0]


# Define a Naive Bayes
clfNB = GaussianNB()

  
# train the classifiers                       
clfNB.fit(x_train, y_train)


#test the trained model on the test set
y_test_pred_NB = clfNB.predict(x_test)


# confusion matrix
confMatrixTestNB = confusion_matrix(y_test, y_test_pred_NB, labels=None)

print ('Conf matrix Naive Bayes')
print (confMatrixTestNB)


# Measures of performance: Precision, Recall, F1

print ('Tree: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro'))
print ('Tree: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NB, average='micro'))
print ('\n')


pr_y_test_pred_NB = clfNB.predict_proba(x_test)


# ROC curve
fprNB, tprNB, thresholdsNB = roc_curve(y_test, pr_y_test_pred_NB[:, 1])

# plot ROC curve 
# Line width, lw=1
lw = 2
plt.plot(fprNB,tprNB,color = 'black', label = 'Naive Bayes')


plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc = "lower right")
plt.show()

