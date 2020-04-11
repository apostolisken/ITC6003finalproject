# ITC6003finalproject
ITC 6003 course project for Deree MSc Data Science. Machine learning, classification, regression, clustering

It consistists of 4 parts that are the 4 tasks we were asked to perform:

1. Classification: Predicting arrhythmia type (20%)
Source data & description: http://archive.ics.uci.edu/ml/datasets/Arrhythmia
The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify it in
one of the 16 groups.

2. Clustering: Market Segmentation: Unsupervised learning (20%)
Source data & description: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
Discover clusters, evaluate and characterize them.

3. Regression (20%)
It is up to you to choose regression task, but you should inform the instructor and get approval for it.
Indicative data sources: Kaggle.com , https://www.analyticsvidhya.com/ ,
https://github.com/awesomedata/awesome-public-datasets , https://www.openml.org/ .

4. Scaling-up: Predicting buys (20%)
Source data & description: https://2015.recsyschallenge.com/challenge.html Your task is to predict
whether a user will buy a product or not based on his/her online behavior, and in particular his/her
clicks during a session. There are two files:
 yoochoose-clicks.dat - Click events. Each record/line in the file has the following fields:
• Session ID – the id of the session. In one session there are one or many clicks.
• Timestamp – the time when the click occurred.
• Item ID – the unique identifier of the item.
• Category – the category of the item.
 yoochoose-buys.dat - Buy events. Each record/line in the file has the following fields:
• Session ID - the id of the session. In one session there are one or many buying events.
• Timestamp - the time when the buy occurred.
• Item ID – the unique identifier of item.
• Price – the price of the item.
• Quantity – how many of this item were bought.
The Session ID in yoochoose-buys.dat will always exist in the yoochoose-clicks.dat file – the records with
the same Session ID together form the sequence of click events of a certain user during the session. The
session could be short (few minutes) or very long (few hours), it could have one click or hundreds of
clicks. All depends on the activity of the user.

Tasks to perform
1. Build a data set that can be used in classifier to decide whether someone will buy or not.
2. Preprocess the data & perform classification
