import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
# import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('C:/Users/leeminsu/PycharmProjects/strokePrediction/healthcare-dataset-stroke-data.csv', na_values=['Unknown', 'Other'])
dataset=pd.read_csv('C:/Users/leeminsu/PycharmProjects/strokePrediction/healthcare-dataset-stroke-data.csv')

print('####After data curation####')
print(dataset.info())
print("####dataset#### ")
print(dataset.head())
print("####dataset describe####")
print(dataset.describe())
print("####dataset info####")
print(dataset.info())
print("####null data####")
print(dataset.isnull().sum())

Stroke_plot = dataset['stroke'].value_counts().reset_index()
Stroke_plot.columns = ['stroke', 'count']

# px.pie(Stroke_plot, values='count', names='stroke', template='plotly', title='Stroke')

# show rate of stroke
plt.figure(figsize=(7, 7))
sns.countplot(x=dataset['stroke'])
plt.title('Rate of stroke', fontsize=20)
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.show()

# show rate of gender
plt.figure(figsize=(7, 7))
sns.countplot(x=dataset['gender'])
plt.title('Rate of gender', fontsize=20)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# drop id column in temporary data
temp_data = dataset.dropna()
# temp_data = temp_data.drop(['id'], axis=1)

# separate target column
x = pd.get_dummies(temp_data.drop(['stroke'], axis=1))
y = temp_data['stroke']

# show feature score
bestfeature = SelectKBest(f_classif, k='all')
fit = bestfeature.fit(x, y)
dfcolumns = pd.DataFrame(x.columns)
dfscores = pd.DataFrame(fit.scores_)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['feature', 'Score']
print("####FeatureScore####")
print(featureScores.nlargest(60, 'Score'))

# correlation hour-per-week with other feature
plt.figure(figsize=(12, 8))
sns.heatmap(temp_data.corr(), linecolor='white', linewidths=1, annot=True)
plt.show()

# find outlier
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].boxplot(dataset['age'])
ax[0].set_title("age")
ax[1].boxplot(dataset['avg_glucose_level'])
ax[1].set_title("avg_glucose_level")
plt.show()

fig = plt.figure(figsize=(7,7))
graph = sns.scatterplot(data=dataset, x='age', y='bmi', hue='gender')
graph.axhline(y=25, linewidth=4, color='r', linestyle='--')

print("####Before drop null####")
print(dataset.isnull().sum())
dataset.dropna(inplace=True)
print("")
print("####After drop null####")
print(dataset.isnull().sum())

dataset = dataset.drop(['id', 'hypertension'], axis=1)
print("")
print("####After drop unnecessary columns####")
print(dataset.info())

indexNames = dataset[(dataset['gender'] != 'Male')
               & (dataset['gender'] != 'Female')].index
dataset.drop(indexNames, inplace=True)
print("####After drop invalid gender####")
print(dataset)

g=sns.FacetGrid(dataset, col='stroke')
g.map(plt.hist, 'age', bins=20)
plt.show()

g=sns.FacetGrid(dataset, col='stroke')
g.map(plt.hist, 'hypertension', bins=20)
plt.show()

X = dataset.drop(['stroke'], axis=1)
y = dataset.pop('stroke')

# function label encoding
# input target column list and dataframe
def lblEncoding(listObj, x):
    lbl = preprocessing.LabelEncoder()

    for i in range(len(listObj)):
        x[listObj[i]] = lbl.fit_transform(x[listObj[i]])
    # output encoded dataframe
    return x

# function ordinal encoding
# input target column list and dataframe
def ordEncoding(listObj, x):
    ord=preprocessing.OrdinalEncoder()

    for i in range(len(listObj)):
        tempColumn=x[listObj[i]].to_numpy().reshape(-1, 1)
        tempColumn=ord.fit_transform(tempColumn)
        tempColumn=tempColumn.reshape(1, -1)[0]
        x[listObj[i]].replace(x[listObj[i]].tolist(), tempColumn, inplace=True)
    # output encoded dataframe
    return x

# function ohehot encoding
# input dataframe
def ohEncoding(x):
    # output encoded dataframe
    return pd.get_dummies(x)

def dtClassifier(trainSetX, trainSetY, testSetX, testSetY):
    dt=DecisionTreeClassifier()
    dt.fit(trainSetX, trainSetY)
    print(dt.score(testSetX, testSetY))

def rfClassifier(trainSetX, trainSetY, testSetX, testSetY):
    rf=RandomForestClassifier()
    rf.fit(trainSetX, trainSetY)
    print(rf.score(testSetX, testSetY))

def knnClassifier(trainSetX, trainSetY, testSetX, testSetY):
    knn=KNeighborsClassifier()
    knn.fit(trainSetX, trainSetY)
    print(knn.score(testSetX ,testSetY))

# function make preprocessing combination
# this function show best accuracy, encoder and scaler for classifiers
# user can search best combination for each classifier by using encoders and scalers
# user have to input data(x), target(y), list of encoder and scaler that user selected
def makeCombination(x, y, encoderList, scalerList, classifierList):
    # user have to input column's name in listObj that have categorical data
    listObj=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    nameEncTemp = ['LabelEncoder', 'OrdinalEncoder', 'OneHotEncoder']
    nameEnc=[]
    checkEnc=[0, 0, 0]
    encoderTemp = [lblEncoding(listObj, x), ordEncoding(listObj, x), ohEncoding(x)]
    encoder=[]
    nameScTemp = ['StandardScaler', 'RobustScaler', 'MaxAbsScaler', 'MinMaxScaler']
    nameSc=[]
    checkSc=[0, 0, 0, 0]
    scalerTemp = [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.MaxAbsScaler(),
              preprocessing.MinMaxScaler()]
    scaler=[]
    listDf=[]
    listBestDf=[]
    listClassifier=classifierList

    # find what encoder selected
    for i in range(len(encoderList)):
        enc=encoderList[i]
        for j in range(len(nameEncTemp)):
            if(enc==nameEncTemp[j]):
                checkEnc[j]=1
    # find what scaler selected
    for i in range(len(scalerList)):
        sc=scalerList[i]
        for j in range(len(nameScTemp)):
            if(sc==nameScTemp[j]):
                checkSc[j]=1

    # modify encoder list according to user's choice
    for i in range(len(checkEnc)):
        if(checkEnc[i]==1):
            nameEnc.append(nameEncTemp[i])
            encoder.append(encoderTemp[i])
    # modify scaler list according to user's choice
    for i in range(len(checkSc)):
        if(checkSc[i]==1):
            nameSc.append(nameScTemp[i])
            scaler.append(scalerTemp[i])

    # make dataframe(number of encoder X number of scaler) of each combination and store in listDf
    for i in range(len(encoder)):
        tempX = encoder[i]
        col=tempX.columns.values
        for j in range(len(scaler)):
            sc=scaler[j]
            tempX=sc.fit_transform(tempX)
            tempX=pd.DataFrame(tempX, columns=col)
            listDf.append(tempX)

    # search best encoder and scaler for each classifier
    for i in range(len(listClassifier)):
        classifer=listClassifier[i]
        scoreMax=0
        indexMax=0
        encBest=''
        scBest=''
        print(classifer)
        for j in range(len(listDf)):
            trainSetX, testSetX, trainSetY, testSetY = train_test_split(listDf[j], y, test_size=0.2)
            classifer.fit(trainSetX, trainSetY)
            score=classifer.score(testSetX, testSetY)
            print(score)
            if(scoreMax<=score):
                scoreMax=score
                indexMax=j
        # store best dataframe in list
        listBestDf.append(listDf[indexMax])
        print('####Function result####')
        print('Best accuracy :', scoreMax)
        encBest=nameEnc[(int)(indexMax/4)]
        scBest=nameSc[indexMax%4]
        print('Best combination : Encoding -> ', encBest, '  Scaling -> ', scBest, '\n')

    # user can get dataframe encoded and scaled with encoder and scaler that make best accuracy
    # there are three dataframe in listBestDf
    # listBestDf[0] is dataframe for decision tree
    # listBestDf[1] is dataframe for random forest
    # listBestDf[2] is dataframe for k neighbors
    # output list of best dataframe
    return listBestDf

# function evaluate each model
# input data and target, classifier
def evaluation(x, y, classifier):
    yy=pd.Series(np.arange(0, 4908)%2)
    print(yy)
    trainSetX, testSetX, trainSetY, testSetY = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=yy)
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    score=cross_val_score(classifier, trainSetX, trainSetY, cv=skf)
    print(classifier, '\nCross validation score :', score)
    print('Mean score :', score.mean())
    classifier.fit(trainSetX, trainSetY)
    print('Accuracy on test set :', classifier.score(testSetX, testSetY))

    matrix=confusion_matrix(testSetY, classifier.predict(testSetX))
    print('Confusion metrics\n', matrix)
    # compute precision, recall and F1 score
    precision=matrix[0][0]/(matrix[0][0]+matrix[1][0])
    recall=matrix[0][0]/(matrix[0][0]+matrix[0][1])
    print('Precision :', precision)
    print('Recall :', recall)
    print('F1 score :', 2*precision*recall/(precision+recall), '\n')
    print('Classification report\n', classification_report(testSetY, classifier.predict(testSetX)))

# user can select classifier, encoder and scaler to make combination
listClassifier=[DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
listEncoder=['LabelEncoder', 'OneHotEncoder']
listScaler=['StandardScaler', 'RobustScaler', 'MaxAbsScaler', 'MinMaxScaler']
listBestDf=makeCombination(X, y, listEncoder, listScaler, listClassifier)
for i in range(len(listBestDf)):
    # evaluate before GridSearchCV
    # listBestDf index 0=DecisionTree, 1=RandomForest, 2=KNeighbors
    evaluation(listBestDf[i], y, listClassifier[i])

# grid dt
# search max features and max depth
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[0], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(testSetX.columns)), 'max_depth': np.arange(1, 20)}]
dt_gscv = GridSearchCV(listClassifier[0], param_grid, cv=5)
dt_gscv.fit(trainSetX, trainSetY)
print(dt_gscv.best_params_)
print('Best score :', dt_gscv.best_score_)

# grid rf
# search max features and max depth
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[1], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(testSetX.columns)), 'max_depth': np.arange(1, 10)}]
rf_gscv = GridSearchCV(listClassifier[1], param_grid, cv=5, n_jobs=-1)
rf_gscv.fit(trainSetX, trainSetY)
print(rf_gscv.best_params_)
print('Best score :', rf_gscv.best_score_)

# grid knn
# search n neighbors
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[2], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'n_neighbors': np.arange(1, 10)}]
knn_gscv = GridSearchCV(listClassifier[2], param_grid, cv=5, n_jobs=-1)
knn_gscv.fit(trainSetX, trainSetY)
print(knn_gscv.best_params_)
print('Best score :', knn_gscv.best_score_)

print('\n---------After GridSearchCV---------\n')
# directly entered the obtained hyper parameter value
dt=DecisionTreeClassifier(max_depth=3, max_features=8)
rf=RandomForestClassifier(max_depth=9, max_features=8)
knn=KNeighborsClassifier(n_neighbors=6)
# evaluate after GridSearchCV
evaluation(listBestDf[0], y, dt)
evaluation(listBestDf[1], y, rf)
evaluation(listBestDf[2], y, knn)

