![](https://user-images.githubusercontent.com/1590890/59088793-1763c100-893b-11e9-87b3-391f507522d9.jpg)

[Feature selection visualization](https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization/notebook)

```python
# 1. count plot
y = data.diagnosis 
ax = sns.countplot(y,label="Count")       # M = 212, B = 357
B, M = y.value_counts()

# Before violin and swarm plot we need to normalization or standirdization
# violinplot - first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
# - in texture_mean feature, median of the Malignant and Benign looks like separated so it can be good for classification. 
# - However, in fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification

# 2. box plots are also useful in terms of seeing outliers
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)

# - variable of concavity_worst and concave point_worst looks like similar but how can we decide whether they are correlated with
# - each other or not. (Not always true but, basically if the features are correlated with each other we can drop one of them)
# - In order to compare two features deeper, lets use joint plot
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")

# What about three or more feauture comparision ? For this purpose we can use pair grid plot
sns.set(style="white")
df = x.loc[:,['radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)

# 3. swarm plot
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)



```

