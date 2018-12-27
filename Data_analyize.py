#1. Data Load and preprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from pandas import Series

file = './word_v1_retention.csv'
with open(file,'r',encoding='utf-8') as f:
    data = pd.read_csv(f,sep='|',parse_dates=['install_date','first_iap_date','retention_ts'])

# remove duplicate user_id records
data_drop_1 = data.drop_duplicates(subset='user_id', keep=False)
print(data_drop_1.shape)

# remove incorrect level_max records
data_drop_2 = data_drop_1.drop(data_drop_1[data_drop_1.level_max >= 4000].index)
print(data_drop_2.shape)
data_drop_2.loc[data_drop_2['level_max'].isna(), 'level_max'] = 0

# remove unused data
data_drop_3 = data_drop_2.drop(data_drop_2[data_drop_2.load_days >= 8].index)
print(data_drop_3.shape)

data_drop_4 = data_drop_3.copy()
data_drop_4.loc[data_drop_4['coin_after'] > 50000, 'coin_after'] = 50000
data_drop_4.loc[data_drop_4['coin_after'] < 0, 'coin_after'] = 0

data_drop_5 = data_drop_4.copy()
data_drop_5['first_nation'] = data_drop_5['first_nation'].fillna('unknown')
data_drop_5['coin_after'] = data_drop_5['coin_after'].fillna(200)
data_drop_5['enjoy_status'] = data_drop_5['enjoy_status'].apply(lambda x: -1 if pd.isnull(x) else 1 if x == 'enjoy-yes' else 0)
data_drop_5['iap_status'] = data_drop_5['first_iap_date'].apply(lambda x: 0 if pd.isnull(x) else 1)
data_drop_5['retention_status'] = data_drop_5['retention_ts'].apply(lambda x: 0 if pd.isnull(x) else 1)

# numeric platform
data_drop_5.drop(data_drop_5[data_drop_5['platform'] == 'NaN'].index, inplace=True)
data_drop_5['platform_num'] = data_drop_5.iloc[:, 2].apply(lambda x: 0 if x == 'googleplay' else 1)

data_changed = data_drop_5.copy()
data_changed = data_changed.drop(columns=['first_iap_date','retention_ts'])


data_load = data_changed[data_changed.load_days >= 3]

print(data_load.retention_status.value_counts())

#2. Util func
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

def plot_cv_result(results, scoring):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 20)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_max_depth'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()
    
# use cv_results_ as parameter
def gs_report(cv_results_, n_top=10):
    """Report top n paramters settings
    """
    means = cv_results_['mean_test_Accuracy']#['mean_test_score']
    stds = cv_results_['std_test_Accuracy']#['std_test_score']

#     for mean, std, params in zip(means, stds, cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#             % (mean, std * 2, params))

    top_scores = sorted(dict(zip(means, cv_results_['params'])).items(), 
                       key = itemgetter(0),
                       reverse = True)[:n_top]

    # convert list of tuples to dict
    return dict(top_scores)

# use grid_scores_ as parameter, but it is deprecated after sklearn 0.20
def report(grid_scores, n_top=10):
    """Report top n paramters settings
    """
    top_scores = sorted(grid_scores, 
                       key=itemgetter(1),
                       reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    
    return top_scores[0].parameters

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.
    """
    
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    
    grid_search = GridSearchCV(clf,
                               param_grid = param_grid,
                               scoring = scoring,
                               refit = 'AUC',
                               return_train_score=True,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.cv_results_))) #grid_scores_)))

#     plot_cv_result(grid_search.cv_results_, scoring)

    print(grid_search.cv_results_.keys())

    top_params = gs_report(grid_search.cv_results_, 10)
    return  top_params

def plot_roc(res):
    fpr, tpr, _ = res
    auc_data = auc(fpr, tpr)

    fig = plt.figure(figsize = (6, 6), dpi = 80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(fpr, tpr, 'b', auc_data)

    legend = plt.legend(loc = 4, shadow = True)
    plt.show()

# decision tree cross validation
from operator import itemgetter
from time import time

from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz


train_data, test_data = train_test_split(data_load, test_size = 0.2, random_state = 100)

fs = [6, 7, 8, 9, 10, 11, 12, 14]
X = train_data.iloc[:, fs].values
y = train_data.iloc[:, 13].values
X_test = test_data.iloc[:, fs].values
y_test = test_data.iloc[:, 13].values

dt_old = DecisionTreeClassifier()
dt_old.fit(X, y)

scores = cross_val_score(dt_old, X, y, cv=5)
print("Without parameters mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                        scores.std()),
                                        end="\n\n" )

print("Grid Parameter Search via 3-fold CV")
    
param_grid = {#"criterion": ["gini", "entropy"],
            #"min_samples_split": range(2, 60, 10),#[5, 6,  10, ],
            "max_depth": range(4, 10, 1),#[5, 8, 10],
            # "min_samples_leaf": [5, 10, 20],
            "max_leaf_nodes": range(10, 160, 10),#[10, 60, 70, 80, 90, 100, 120],
            }
    
dt = DecisionTreeClassifier()
ts_gs = run_gridsearch(X, y, dt, param_grid, cv=5)
    
print("Best Param:")
for k, v in ts_gs.items():
    print("parameter: {} setting: {}".format(k, v))

# test with best parameters
print("Testing best parameters Grid ...")
print(list(ts_gs.values())[0])
dt_ts_gs = DecisionTreeClassifier(**list(ts_gs.values())[0])

dt_ts_gs.fit(X, y)

dt_grap = export_graphviz(dt_ts_gs, out_file=None)
graphviz.Source(dt_grap)

# scores = cross_val_score(dt_ts_gs, X, y, cv=3)
# print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()),
#                                               end="\n\n" )

y_pred = dt_ts_gs.predict(X_test)

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted Non-return Users', 'Predicted Return Users'],
    index=['True Non-return Users', 'True Return Users']
))

dt_grap = export_graphviz(dt_ts_gs, 
                          feature_names=data_load.iloc[:, fs].columns,
                          class_names=True,
                          filled=True,
                          rounded=True,
                          special_characters=True,
                          out_file=None)

graphviz.Source(dt_grap)

# with more params
print("Grid Parameter Search via 5-fold CV")
    
param_grid = {#"criterion": ["gini", "entropy"],
            "min_samples_split": range(50, 200, 50),
            "max_depth": range(4, 8, 1),
            "min_samples_leaf": [100, 300, 400],
            # "max_features": ["sqrt"],
            "max_leaf_nodes": range(20, 90, 10),
            }
    
dt = DecisionTreeClassifier()
ts_gs = run_gridsearch(X, y, dt, param_grid, cv=5)
    
print("Best Param:")
for k, v in ts_gs.items():
    print("parameter: {} setting: {}".format(k, v))

# test with best parameters
print("Testing best parameters Grid ...")
print(list(ts_gs.values())[0])
dt_ts_gs = DecisionTreeClassifier(**list(ts_gs.values())[0])

dt_ts_gs.fit(X, y)

# dt_grap = export_graphviz(dt_ts_gs, out_file=None)
# graphviz.Source(dt_grap)

y_pred = dt_ts_gs.predict(X_test)

print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted Non-return Users', 'Predicted Return Users'],
    index=['True Non-return Users', 'True Return Users']
))

dt_grap = export_graphviz(dt_ts_gs, 
                          feature_names = data_load.iloc[:, fs].columns,
                          class_names = True,
                          filled = True,
                          rounded = True,
                          special_characters = True,
                          out_file = None)

graphviz.Source(dt_grap)

# Decistion tree + Logistic Regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression 

m = 5
features = ['level_max', 'load_days', 'coin_after', 'challenge_gids', 'challenge_games', 'enjoy_status', 'iap_status']
label = ['retention_status']

def dt_log(m, features, label):
    trainDT, testDT = train_test_split(data_load, test_size=0.2, random_state=1)
#     trainDT, cvDT = train_test_split(trainDT, test_size=0.2, random_state=1)

    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(trainDT[features[:m]], trainDT[label])

    leaf = dt.apply(trainDT[features[:m]])
    leafNode = leaf.reshape(-1, 1)
    
    coder = OneHotEncoder()
    coder.fit(leafNode)

    newFeature = np.c_[
        coder.transform(dt.apply(trainDT[features[:m]]).reshape(-1, 1)).toarray(),
        trainDT[features[m:]]]
    logit = LogisticRegression()
    logit.fit(newFeature[:, 1:], trainDT[label].values.ravel())
    
    testFeature = np.c_[
        coder.transform(dt.apply(testDT[features[:m]]).reshape(-1, 1)).toarray(),
        testDT[features[m:]]]
    y_predprob = logit.predict_proba(testFeature[:, 1:])
    y_pred = np.argmax(y_predprob, axis=1)

    print(confusion_matrix(testDT[label]['retention_status'].values, y_pred))
    print("Accuracy : %.4g" % accuracy_score(testDT[label]['retention_status'].values, y_pred))
    print("AUC Score (Test): %f" % roc_auc_score(testDT[label]['retention_status'].values, y_predprob[:, 1]))    
    
    res = roc_curve(testDT[label], y_predprob[:, 1])
    plot_roc(res)


dt_log(m, features, label)

# Plot learning curve
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve( 
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    return plt

#GBT
'''
With multiple learning rate
'''
def gbt(X_train, y_train, X_test, y_test):
    learning_rates = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # TODO: try more estimators...
    n_estimators_ =[10,100,300,500,1000,2000,5000]
    for learning_rate in learning_rates:
        gb = GradientBoostingClassifier(n_estimators=30, learning_rate = learning_rate, max_features=2, 
                                        max_depth = 10, random_state = 0)
        gb.fit(X_train, y_train)
        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
        print()    

'''
With GridSearchCV
'''
def gbt_cv(X_train, y_train):
    parameters = {
        "loss":["deviance"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15],
        # "min_samples_split": np.linspace(0.1, 0.5, 12),
        # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth":[3,5,8],
        "max_features":["log2","sqrt"],
        "criterion": ["friedman_mse",  "mae"],
        "subsample":[0.5, 0.8, 0.9, 1.0],
        "n_estimators":[30]
        }

    clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1)

    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.best_params_)

# test codes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

train_data, test_data = train_test_split(data_load, test_size = 0.2, random_state = 100)

def select_feature(fs):
    X_train = train_data[fs].values
    y_train = train_data.iloc[:, 13].values

    X_test = test_data[fs].values
    y_test = test_data.iloc[:, 13].values

    return [X_train, y_train, X_test, y_test]

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'enjoy_status', 'iap_status']
X_train, y_train, X_test, y_test = select_feature(fs)

print("Train result with feature 6 ~ 12")
gbt(X_train, y_train, X_test, y_test)

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'enjoy_status', 'iap_status', 'platform_num']
X_train, y_train, X_test, y_test = select_feature(fs)

print("Train result with feature 6 ~ 12 with platform")
gbt(X_train, y_train, X_test, y_test)

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'iap_status', 'platform_num']
X_train, y_train, X_test, y_test = select_feature(fs)

print("Train result with feature 6 ~ 12 without enjoy_status")
gbt(X_train, y_train, X_test, y_test)


# run with better learning rate 0.1 relately
print('Test GBT with learning rate 0.1')

gbt = GradientBoostingClassifier(n_estimators=30, learning_rate = 0.1, max_features=2, 
                                max_depth = 10, random_state = 0)
gbt.fit(X_train, y_train)

print("Result: ")
print("Accuracy score (training): {0:.3f}".format(gbt.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gbt.score(X_test, y_test)))

y_pred = gbt.predict(X_test)

print(confusion_matrix(y_test, y_pred))

res = roc_curve(y_pred, y_test)
plot_roc(res)

# Try more methods to validate model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def modelfit(clf, train_data, test_data, features, performCV = True, printFeatureImportance = True, cv_folds = 5):
    # Split train data and test data by features
    X_train = train_data[features].values
    y_train = train_data.iloc[:, 13].values
    X_test = test_data[features].values
    y_test = test_data.iloc[:, 13].values
    
    # Fit the model
    clf.fit(X_train, y_train)

    # Predict training data
    train_pred = clf.predict(X_train)
    train_predprob = clf.predict_proba(X_train)[:,1]
    
    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(clf, X_train, y_train, cv = cv_folds, scoring = 'roc_auc')
    
    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % accuracy_score(y_train, train_pred))
    print("AUC Score (Train): %f" % roc_auc_score(y_train, train_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),
                                                                                 np.std(cv_score),
                                                                                 np.min(cv_score),
                                                                                 np.max(cv_score)))
        
    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(clf.feature_importances_, features).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    # Prediction test data
    y_pred = clf.predict(X_test)
    y_predprob = clf.predict_proba(X_test)[:,1]
        
    # Print confusion matrix
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=['Predicted Non-return Users', 'Predicted Return Users'],
        index=['True Non-return Users', 'True Return Users']
        ))

gbt = GradientBoostingClassifier(n_estimators=10, learning_rate = 0.1, max_features = 3, 
                                max_depth = 8, random_state = 0)

train_data, test_data = train_test_split(data_load, test_size = 0.2, random_state = 100)

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'enjoy_status', 'iap_status']
modelfit(gbt, train_data, test_data, fs)

# GBTs with GridSearchCV
param_test1 = {'n_estimators':range(20,81,10)}

gs1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                          min_samples_split=500,
                                                          min_samples_leaf=50,
                                                          max_depth=8,
                                                          max_features='sqrt',
                                                          subsample=0.8,
                                                          random_state=10), 
                   param_grid = param_test1, 
                   scoring='roc_auc',
                   n_jobs=4, 
                   iid=False, 
                   cv=5)

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'enjoy_status', 'iap_status']
X_train = train_data[fs].values
y_train = train_data.iloc[:, 13].values

gs1.fit(X_train, y_train)

gs1.grid_scores_, gs1.best_params_, gs1.best_score_

# XGBoost
# XGBoot with gridsearchCV
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

fs = ['level_max', 'load_days', 'challenge_gids', 'challenge_games', 'coin_after', 'iap_status']
X_train, y_train, X_test, y_test = select_feature(fs)

parameters = {'nthread':[4], # when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05, 0.1],
              'max_depth': [4, 6, 8],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [100], # number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

xgb = XGBClassifier()

clf = GridSearchCV(xgb, parameters, 
                   n_jobs=5, 
                   cv=3, 
                   scoring='roc_auc',
                   verbose=2, 
                   refit=True)

clf.fit(X_train, y_train)

''' check grid_scores_'''
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('xgboost with gridsearchcv score:', score)

# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

# test_probs = clf.predict_proba(X_test)[:,1]

''' check cv_results_ '''
# print('xgb cv result: ')
# print(clf.cv_results_)

# print('xgb Best estimator:')
# print(clf.best_estimator_)

print('xgb best score: ')
print(clf.best_score_)

print('Best hyperparameters:')
print(clf.best_params_)

best_est = clf.best_estimator_
print(best_est)

# run roc
y_pred = best_est.predict(X_test)#predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))  

res = roc_curve(y_pred, y_test)
plot_roc(res)

# learning curve
from sklearn.model_selection import ShuffleSplit
import warnings

# filter the deprecated warning...
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = XGBClassifier(clf.best_params_)

title = "Learning Curves XGBoost"
X = data_load[features].values
y = data_load.iloc[:, 13].values

plot_learning_curve(best_est, title, X, y, (0.71, 0.75), cv=cv, n_jobs=4)

plt.show()









