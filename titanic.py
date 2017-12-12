from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
import xgboost as xgb
import csv
from datetime import datetime
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np

'''
def load_data(file, type):
    try:
        reader = csv.reader(open(file, 'r'))
    except IOError:
        print 'file not exist!'

    X = []
    y = []
    id = []

    for line in reader:
        if reader.line_num == 1:
            continue

        cust_id = int(line[0])
        if type == 'train':
            label = int(line[-1])
        else:
            label = -1
        pclass = int(line[1])
        sex =  0 if line[3] == 'female' else 1
        if line[4] == '':
            age = -1
        else:
            age = float(line[4])
        sib = int(line[5])
        parch = int(line[6])
        fare = float(line[8])
        if line[10] == 'S':
            embarked = 1
        elif line[10] == 'C':
            embarked = 2
        else:
            embarked = 3

        feature = []
        feature.append(pclass)
        feature.append(sex)
        feature.append(age)
        feature.append(sib)
        feature.append(parch)
        feature.append(fare)
        feature.append(embarked)

        X.append(feature)
        y.append(label)
        id.append(cust_id)

    X = MinMaxScaler().fit_transform(X)
    if type == 'train':
        return X, y
    else:
        return id, X
'''

def load_data_new(file, type):
    try:
        reader = pd.read_csv(file)
    except IOError:
        print 'file not exist!'
    if type == 'train':
        X = reader.iloc[:, 1:-1]
        y = reader['Survived']
    else:
        X = reader.iloc[:, 1:]
    id = reader['PassengerId']

    # print X[:5]

    # print X.Age
    # print X.loc[1,'Age']

    print '--------------------------------feature factoring---------------------'
    #dummies = pd.get_dummies(X.loc[:, ['Pclass', 'Sex', 'Cabin', 'Embarked']], prefix=['Pclass', 'Sex', 'Cabin', 'Embarked'], dummy_na=True)
    X.loc[pd.notnull(X['Cabin']), 'Cabin'] = 'yes'
    X.loc[pd.isnull(X['Cabin']), 'Cabin'] = 'no'

    # print X[:10]

        # if X.loc[i, 'Age'] <= 12:
        #     child.append(1)
        # else:
        #     child.append(0)

        # if X.loc[i,'Title'] ==0 and X.loc[i,'Parch'] > 1:
        #     mother.append(1)
        # else:
        #     mother.append(0)
    print '-----------------------------feature engineering-------------------------'
    title =[]
    for name in X.loc[:,'Name']:
        # must caution the order of 'Mrs' and 'Mr'!!!
        if 'Mrs' in name:
            title.append('Mrs')
        elif 'Mr' in name:
            title.append('Mr')
        elif 'Miss' in name:
            title.append('Miss')
        else:
            title.append(np.nan)

    X['Title'] = title

    # X.loc['Mrs' in X['Name'], 'Title'] = 'Mrs'
    # X.loc['Mr' in X['Name'], 'Title'] = 'Mr'
    # X.loc['Miss' in X['Name'], 'Title'] = 'Miss'

    # fill the missing value of age using the mean value of corresponding title
    for i in range(len(X)):
        if pd.isnull(X.loc[i, 'Age']):
            X.loc[i, 'Age'] = X[X.Title==X.loc[i,'Title']].loc[:,'Age'].mean()

    # print 'the missing value of age is:'
    # print X['Age'].isnull().sum()

    X.loc[pd.isnull(X['Age']), 'Age'] = X['Age'].mean()

    # print 'the missing value of age is:'
    # print X['Age'].isnull().sum()

    # create a new feature Child
    X['Child'] = 0
    X.loc[X['Age'] <= 12, 'Child'] = 1

    # create a new feature Mother, pythonic code!!!
    X['Mother'] = 0
    X.loc[[a and b for a, b in zip(X['Title']=='Mrs', X['Parch'] > 1)], 'Mother'] = 1

    # print X.loc[X['Age'].isnull()]
    # print X.ix[65,'Age']

    X['Age'] = [int(age) / 10 for age in X['Age']]

    # print X[:10]

    print '---------------------------------dummy--------------------------------'
    title_dummy = pd.get_dummies(X['Title'], prefix='Title', dummy_na=True)
    pclass_dummy = pd.get_dummies(X['Pclass'], prefix='Pclass', dummy_na=True)
    sex_dummy = pd.get_dummies(X['Sex'], prefix='Sex', dummy_na=True)
    cabin_dummy = pd.get_dummies(X['Cabin'], prefix='Cabin', dummy_na=True)
    embarked_dummy = pd.get_dummies(X['Embarked'], prefix='Embarked', dummy_na=True)
    age_dummy = pd.get_dummies(X['Age'], prefix='Age_bin')

    X = pd.concat([X, title_dummy, pclass_dummy, sex_dummy, cabin_dummy, embarked_dummy, age_dummy], axis=1)


    X.drop(['Title', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace = True)

    # print X[:10]
    print X.columns


    X = MinMaxScaler().fit_transform(X)

    if type == 'train':
        return X, y
    else:
        return id, X


def train_model(X_train, y_train):
    xg = xgb.XGBClassifier(nthread=-1, n_estimators=50, max_depth=15, learning_rate=0.1, subsample=1.0, colsample_bytree=0.8, gamma=1)
    '''
    grid = GridSearchCV(xg, param_grid={'max_depth': [6, 10, 15],
                                        'learning_rate': [0.1, 1],
                                        'gamma': [1, 2, 5],
                                        'subsample': [0.8, 1.0],
                                        'colsample_bytree': [0.8, 1.0]}, scoring='accuracy', cv=3)
    print grid.grid_scores_
    print grid.best_score_, grid.best_params_
    # grid.fit(X_train, y_train)
    '''

    xg.fit(X_train, y_train)

    return xg


def test_model(X_test, y_test, clf):
    pre = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print 'the score is: '+ str(score)


def submit(test_file, submit_file, clf):
    id, X =load_data_new(test_file, 'test')
    print X.shape
    pre = clf.predict(X)
    solutions = []
    for i in range(len(id)):
        solutions.append([id[i], pre[i]])

    writer = csv.writer(open(submit_file, 'w'))
    writer.writerow(['PassengerId', 'Survived'])
    writer.writerows(solutions)


def main():
    train_file = 'train.csv'
    test_file = 'test.csv'
    submit_file = 'submit.csv'

    print '----------------load data----------------'
    X, y = load_data_new(train_file, 'train')
    print 'the shape of sample is:'
    print X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print '----------------train model--------------'
    start_time = datetime.now()
    clf = train_model(X_train, y_train)
    end_time = datetime.now()
    print 'the time for xgboost: %s seconds' % (end_time - start_time).seconds

    print '----------------test model---------------'
    test_model(X_test, y_test, clf)

    print '----------------submit-------------------'
    submit(test_file, submit_file, clf)


if __name__ == '__main__':
    main()