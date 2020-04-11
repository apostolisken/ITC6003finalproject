# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:25:21 2020

@author: apost
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:45:06 2020

@author: apost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from ml_metrics import rmse
from ml_metrics import mae
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def scores(actuals, predicteds):
    rmses = rmse(actual = actuals, predicted = predicteds)
    mses = mean_squared_error(actuals,predicteds)
    maes = mae(actual = actuals, predicted = predicteds)
    r2s = r2_score(actuals,predicteds)
    return rmses,maes,r2s,mses

#import the dataset
data = pd.read_csv('FuelConsumptionCo2.csv')
#keep the lines that play a role in co2emissions
cols = ['ENGINESIZE','CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']
df = data[cols]

print(df['FUELTYPE'].unique())

#factorize, like dummies but in a single column, converts letter categories to number categoriess
df['FUELTYPE'] = pd.factorize(df['FUELTYPE'])[0]
df['FUELTYPE'] +=1 #because we need no zeros for the log transformation

print(df['FUELTYPE'].unique())

#log transformation to increase accuracy
df = np.log(df)

print()
print()

#------------------------------ Simple Linear Regression -------------------
print('------------------Simple Linear Regression: ')
print()
#linear model for how engine size affects co2 emmissions
model = linear_model.LinearRegression()

xtrain,xtest,ytrain,ytest = train_test_split(df[['ENGINESIZE']],df[['CO2EMISSIONS']],test_size = 0.3)

model.fit(xtrain,ytrain)

SR_trainpred = np.array(model.predict(xtrain))
SR_testpred = np.array(model.predict(xtest))

print('Regression Line for engine size / CO2 emissions:')

#regression line for the predicted in test
plt.scatter(xtest, ytest,  color='maroon')
plt.plot(xtest, SR_testpred, color='cyan', linewidth=2)
plt.show()
print()

#print metrics for the simple linear regression model

SR_metrics_train = scores(ytrain,SR_trainpred)
SR_metrics_test = scores(ytest, SR_testpred)

print('Train mse: ',SR_metrics_train[3])
print('Test mse: ',SR_metrics_test[3])

print('Train rmse: ',SR_metrics_train[0])
print('Test rmse: ',SR_metrics_test[0])

print('Train mae: ',SR_metrics_train[1])
print('Test mae: ',SR_metrics_test[1])

print('Train R2: ',SR_metrics_train[2])
print('Test R2: ',SR_metrics_test[2])

print()
print()

#---------------------------------Polynomial Regression-----------------------
print('------------Polynomial Regression: ')
print()


#split the dataset for fitting
X_train, X_test, Y_train, Y_test = train_test_split(df[['ENGINESIZE']],df[['CO2EMISSIONS']], test_size=0.3)

#regression model
poly = PolynomialFeatures(degree=2)
#regr = linear_model.LinearRegression()

polyTR=poly.fit_transform(X_train)
polyTE=poly.fit_transform(X_test)

#train the model
model.fit(polyTR, Y_train)

#predict the values
PR_trainpred = model.predict(polyTR)
PR_testpred = model.predict(polyTE)


#print metrics for Polynomial Regression
PR_metrics_train = scores(Y_train,PR_trainpred)
PR_metrics_test = scores(Y_test, PR_testpred)

print('Train mse: ',PR_metrics_train[3])
print('Test mse: ',PR_metrics_test[3])

print('Train rmse: ',PR_metrics_train[0])
print('Test rmse: ',PR_metrics_test[0])

print('Train mae: ',PR_metrics_train[1])
print('Test mae: ',PR_metrics_test[1])

print('Train R2: ',PR_metrics_train[2])
print('Test R2: ',PR_metrics_test[2])

print()

#regression line for simple and polynomial 
plt.scatter(X_test, Y_test,  color='magenta')
plt.plot(X_test, PR_testpred, '*', color='darkgreen')
plt.show()
print()

#regression line for the predicted in test
plt.scatter(X_test, Y_test,  color='magenta')
plt.plot(X_test, PR_testpred, '*', color='darkgreen')
plt.plot(xtest, SR_testpred, color='cyan', linewidth=2)
plt.show()
print()

#---------------------------------Multiple Linear Regression-----------------------
print('------------------Multiple Linear Regression: ')
print()
#linear model for how all features affect co2 emissions
targetcol = 6
selFeatures = list(df.columns.values)
del selFeatures[targetcol]

xtrainm,xtestm,ytrainm,ytestm = train_test_split(df[selFeatures],df['CO2EMISSIONS'],test_size = 0.3)

model.fit(xtrainm,ytrainm)

MR_trainpred = np.array(model.predict(xtrainm))
MR_testpred = np.array(model.predict(xtestm))


#print metrics for the simple linear regression model

MR_metrics_train = scores(ytrainm,MR_trainpred)
MR_metrics_test = scores(ytestm, MR_testpred)

print('Train mse: ',MR_metrics_train[3])
print('Test mse: ',MR_metrics_test[3])

print('Train rmse: ',MR_metrics_train[0])
print('Test rmse: ',MR_metrics_test[0])

print('Train mae: ',MR_metrics_train[1])
print('Test mae: ',MR_metrics_test[1])

print('Train R2: ',MR_metrics_train[2])
print('Test R2: ',MR_metrics_test[2])

print()

#-----------------------------------------Ridge-Lasso Simple Reg (engine size)
#Check diff with simple
print('------------------Ridge and Lasso Regression: ')

ridgeReg = Ridge(alpha=0.00001)
ridgeReg.fit(xtrain, ytrain)

lassoReg = Lasso(alpha=0.000001, max_iter = 10e5)
lassoReg.fit(xtrain, ytrain)
coeff_used = np.sum(lassoReg.coef_!=0)

RR_trainpred = ridgeReg.predict(xtrain)
RR_testpred = ridgeReg.predict(xtest)

LR_trainpred = lassoReg.predict(xtrain)
LR_testpred = lassoReg.predict(xtest)

RR_metrics_train = scores(ytrain,RR_trainpred)
RR_metrics_test = scores(ytest, RR_testpred)

LR_metrics_train = scores(ytrain,LR_trainpred)
LR_metrics_test = scores(ytest, LR_testpred)

#print Ridge

print('Train mse Ridge: ',RR_metrics_train[3])
print('Test mse Ridge: ',RR_metrics_test[3])

print('Train rmse Ridge: ',RR_metrics_train[0])
print('Test rmse Ridge: ',RR_metrics_test[0])

print('Train mae Ridge: ',RR_metrics_train[1])
print('Test mae Ridge: ',RR_metrics_test[1])

print('Train R2 Ridge: ',RR_metrics_train[2])
print('Test R2 Ridge: ',RR_metrics_test[2])
print()

#print Lasso

print('Train mse Lasso: ',LR_metrics_train[0])
print('Test mse Lasso: ',LR_metrics_test[0])

print('Train rmse Lasso: ',LR_metrics_train[1])
print('Test rmse Lasso: ',LR_metrics_test[1])

print('Train mae Lasso: ',LR_metrics_train[1])
print('Test mae Lasso: ',LR_metrics_test[1])

print('Train R2 Lasso: ',LR_metrics_train[2])
print('Test R2 Lasso: ',LR_metrics_test[2])
print('Number of features used: ', coeff_used)

print()
print()


#-----------------------------------------Ridge-Lasso Multiple Reg (all features)
#Check diff with multiple
print('------------------Ridge and Lasso Regression Multiple: ')

ridgeReg = Ridge(alpha=0.00001)
ridgeReg.fit(xtrainm, ytrainm)

lassoReg = Lasso(alpha=0.000001, max_iter = 10e5)
lassoReg.fit(xtrainm, ytrainm)
coeff_used2 = np.sum(lassoReg.coef_!=0)

RR_trainpred = ridgeReg.predict(xtrainm)
RR_testpred = ridgeReg.predict(xtestm)

LR_trainpred = lassoReg.predict(xtrainm)
LR_testpred = lassoReg.predict(xtestm)

RR_metrics_train = scores(ytrainm,RR_trainpred)
RR_metrics_test = scores(ytestm, RR_testpred)

LR_metrics_train = scores(ytrainm,LR_trainpred)
LR_metrics_test = scores(ytestm, LR_testpred)

#print Ridge

print('Train mse Ridge: ',RR_metrics_train[3])
print('Test mse Ridge: ',RR_metrics_test[3])

print('Train rmse Ridge: ',RR_metrics_train[0])
print('Test rmse Ridge: ',RR_metrics_test[0])

print('Train mae Ridge: ',RR_metrics_train[1])
print('Test mae Ridge: ',RR_metrics_test[1])

print('Train R2 Ridge: ',RR_metrics_train[2])
print('Test R2 Ridge: ',RR_metrics_test[2])
print()

#print Lasso

print('Train mse Lasso: ',LR_metrics_train[0])
print('Test mse Lasso: ',LR_metrics_test[0])

print('Train rmse Lasso: ',LR_metrics_train[1])
print('Test rmse Lasso: ',LR_metrics_test[1])

print('Train mae Lasso: ',LR_metrics_train[1])
print('Test mae Lasso: ',LR_metrics_test[1])

print('Train R2 Lasso: ',LR_metrics_train[2])
print('Test R2 Lasso: ',LR_metrics_test[2])
print('Number of features used: ', coeff_used2)

print()
print()
