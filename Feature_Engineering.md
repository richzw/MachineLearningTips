
![fe](https://user-images.githubusercontent.com/1590890/47628335-19c59780-db70-11e8-9124-4b0169bb369d.png)

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

'''数据离散化'''

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
```
