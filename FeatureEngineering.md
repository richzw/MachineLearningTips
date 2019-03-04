
![fe](https://user-images.githubusercontent.com/1590890/47628335-19c59780-db70-11e8-9124-4b0169bb369d.png)

```python
## Source: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#histogram
sns.distplot(df_train['SalePrice']);

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

##1. Relationship with **numerical variables**
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

##2. Relationship with **categorical features**
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

#3. correlation matrix - fetch n largest
corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

#4. missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1) # delete by column
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) # delete by row
df_train.isnull().sum().max() #just checking that there's no missing data missing...


```


```python
'''
归一化
'''

from sklearn import preprocessing

X_tr = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()

X_tr_minmax = min_max_scaler.fit_transform(X_tr)
print(X_tr_minmax)

'''
feature scaler: 
  1. to make gradient descent more quickly 
  2. possible more accuracy
'''

X_scaled = preprocessing.scale(X_tr)
print(X_scaled)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

'''
数据离散化
'''

arr = np.random.randn(20)
factor = pd.cut(arr, 4) # cut - bins based on values
print(factor)

factor1 = pd.cut(arr, [-5, -1, 0, 1, 5])
print(factor1)

'''
dummy variable

dum_pclass = pd.get_dummies(data_tr['pclass'], prefix='pclass')
'''

# feature selection
# 1. 过滤性，缺点，木有考虑特征之间关联性，可能会把有用的特征去掉

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
X_ir, y_ir = iris.data, iris.target
print(X_ir.shape)

X_new = SelectKBest(chi2, k=2).fit_transform(X_ir, y_ir)
print(X_new.shape)

# 2. 包裹型，特征子空间搜索问题，筛选各种特征子空间，用模型来估计效果。例如 递归特征删除 - per auc
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

boston = load_boston()
X_bo = boston['data']
y_bo = boston['target']
names = boston['feature_names']

lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X_bo, y_bo)

print("Features sorted by rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

'''
组合特征

- 用GBDT产出特征组合路径
- 组合特征和原始特征一起放进LR训练

基于树模型的组合特征：GBDT+LR，每一条分支都可以是一个特征
'''
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

'''
Feature selection

RFE

Given an external estimator that assigns weights to features, recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached

'''
from sklearn.feature_selection import RFECV

# trainDT, testDT = train_test_split(data_load, test_size=0.2, random_state=1)
X = data_load[features]
y = data_load.iloc[:, 13]

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

```

```python
// Missing values
// source: https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

// Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired.
// To deal with this, we'll use the median to impute the missing values.

train_df["Age"].median(skipna=True)


```

