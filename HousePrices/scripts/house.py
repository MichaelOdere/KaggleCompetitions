import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, linear_model, datasets
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import scipy.stats as stats
import os

dataFolder = '../data'
csvFolder = '../data/csv/'
outputFolder = '../output'
predictionsFolder = '../output/predictions/'
def main():
    if not (os.path.isdir(dataFolder)):
        os.mkdir(dataFolder)

    if not (os.path.isdir(csvFolder)):
        os.mkdir(csvFolder)

    if not (os.path.isfile(csvFolder + 'train.csv')):
        print "Missing data can be downloaded at:"
        print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        exit()

    if not (os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    if not (os.path.isdir(predictionsFolder)):
        os.mkdir(predictionsFolder)

    # Read data
    train = pd.read_csv(csvFolder + 'train.csv', encoding='utf-8')
    test = pd.read_csv(csvFolder + 'test.csv', encoding='utf-8')

    # Get all numeric feature columns
    numeric_feats = test.dtypes[test.dtypes != 'object'].index

    # Get all non numeric feature columns
    non_numeric_feats = test.dtypes[test.dtypes == 'object'].index

    # Get skewed features, features that are not normally distributed
    skewed_feats = train[numeric_feats].apply(lambda x: stats.skew(x.dropna())) #compute skewness
    skewed_feats = numeric_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.values

    # Unskew the skewed distribution
    train[skewed_feats] = np.log1p(train[skewed_feats])
    test[skewed_feats] = np.log1p(test[skewed_feats])

    # Input missing data. Used -1 because the fact that data is missing
    # could be a feature itself
    train[numeric_feats] = train[numeric_feats].fillna(-1)
    test[numeric_feats] = test[numeric_feats].fillna(-1)

    # Normalize numeric_feats
    scaler = preprocessing.MinMaxScaler()
    train[numeric_feats] = scaler.fit_transform(train[numeric_feats])
    test[numeric_feats] = scaler.fit_transform(test[numeric_feats])

    # For testing to help find correlatoin between feature and sale price
    numeric_feats = numeric_feats.tolist()

    # TODO: discretize some features
    # person edits to non numeric fts
    # train.MSZoning.replace(['RH'], ['RM'], inplace=True)
    # test.MSZoning.replace(['RH'], ['RM'], inplace=True)

    for column in non_numeric_feats:
        print train.groupby(column).mean().SalePrice.sort_values()

    # Encode a label for non_numeric_feats so they can be processed
    le = preprocessing.LabelEncoder()
    for feat in non_numeric_feats:
        # print train[feat].head()
        train[feat] = le.fit_transform(train[feat])
        test[feat] = le.fit_transform(test[feat])

    # Normalize SalePrice giving it a normal distribution
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # Used for testing with data we have
    # linear_model_functions(train, 1000)

    # Used to create an output file and compare it to another given output file
    make_prediction(train, test)

# Train and test on given data
def linear_model_functions(data, fold):
    train = data[:fold]
    test = data[fold:]
    y_train = train.SalePrice.values.tolist()
    y_test = test.SalePrice.values.tolist()
    train = train.drop('SalePrice', 1)
    test = test.drop('SalePrice', 1)

    x_train = train.values.tolist()
    x_test = test.values.tolist()

    lr = linear_model.LinearRegression()
    lr = lr.fit(x_train, y_train)
    lr_answer = lr.predict(test.values.tolist())
    print "LinearRegression ", rmsle(lr_answer, y_test)

    Lasso = linear_model.LassoCV()
    Lasso = Lasso.fit(x_train, y_train)
    coef = pd.Series(Lasso.coef_, index = train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    lasso_answer = Lasso.predict(test.values.tolist())

    # lasso_answer = np.exp(lasso_answer)
    print "Lasso ", rmsle(lasso_answer, y_test)

# Train on given data test on test data and return predictions in submission.csv
def make_prediction(train, test):

    y_train = train.SalePrice.values.tolist()

    train = train.drop('SalePrice', 1)
    # test = train.drop('Id', 1)
    # train = train.drop('Id', 1)

    x_train = train.values.tolist()
    x_test = test.values.tolist()

    model = linear_model.LassoLarsCV(normalize = False)
    model = model.fit(x_train, y_train)
    answer = model.predict(test.values.tolist())

    df = return_csv_from_arr(answer)
    df.to_csv(predictionsFolder + 'submission.csv' ,  index=False)

def return_csv_from_arr(arr):
    df = pd.DataFrame(arr)
    df = df.reset_index(drop = False)
    df.columns = ['Id', 'SalePrice']
    arr = range(1461, 2920)
    df['Id'] = arr
    df = df.reset_index(drop = True)
    return df

def find_feats(train):
    y_train = train.SalePrice.values.tolist()
    train = train.drop('SalePrice', 1)
    x_train = train.values.tolist()

    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(x_train, y_train)
    print selector.support_
    print selector.ranking_

# Root Mean Squared Logarithmic Error
def rmsle(y, y_pred):
     return np.sqrt((( (np.log1p(y_pred)- np.log1p(y)) )**2).mean())

if __name__ == "__main__":
    main()
