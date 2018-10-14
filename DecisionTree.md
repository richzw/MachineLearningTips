**Decision tree** is a classifier that partitions data recursively into to form groups or classes. 
This is a supervised learning algorithm which can be used in discrete or continuous data for __classification__ or __regression__

--------------------

The Algorithm used in the decision trees are ID3 , C4.5, CART, C5.0, CHAID, QUEST, CRUISE, etc. The splitting of nodes is decided by algorithms like __information gain__, __chi square__, __gini index__.

**ID3**

It builds a decision tree for the given data in a top-down fashion, starting from a set of objects and a specification of properties Resources and Information. each node of the tree, one property is tested based on maximizing information gain and minimizing entropy, and the results are used to split the object set. This process is recursively done until the set in a given sub-tree is homogeneous (i.e. it contains objects belonging to the same category). The ID3 algorithm uses a greedy search. It selects a test using the information gain criterion, and then never explores the possibility of alternate choices.

Disadvantages:

- Data may be over-fitted or over-classified, if a small sample is tested.
- Only one attribute at a time is tested for making a decision.
- Does not handle numeric attributes and missing values.

**C4.5**

Improved version on ID 3 by Quinlan's. The new features (versus ID3) are: 

- accepts both continuous and discrete features; 
- handles incomplete data points; 
- solves over-fitting problem by (very clever) bottom-up technique usually known as "pruning"; 
- different weights can be applied the features that comprise the training data.

Disadvantages:

- constructs empty branches with zero values
- Over fitting happens when algorithm model picks up data with uncommon characteristics , especially when data is noisy.

**Cart**

CART stands for Classification and Regression Trees. It uses Gini Impurity. It is characterized by the fact that it constructs **binary trees**, namely each internal node has exactly two outgoing edges. The splits are selected using the twoing criteria and the obtained tree is pruned by cost–complexity Pruning. CART can handle both numeric and categorical variables and it can easily handle outliers.

Disadvantages:

- It can split on only one variable
- Trees formed may be unstable

---------

Decision Tree implementations differ primarily along these axes:

- the **splitting criterion** (i.e., how "variance" is calculated)
- whether it builds models for **regression** (continuous variables, e.g., a score) as well as **classification** (discrete variables, e.g., a class label)
- technique to eliminate/reduce **over-fitting**
- whether it can handle **incomplete data**

![dt](https://user-images.githubusercontent.com/1590890/46903313-c1eb2780-cf05-11e8-9889-83923c339656.jpeg)

---------


Avoiding Overfittingin Decision Trees
-----

- Stop growing the tree when the data split is not statistically significant
- Grow the full tree, then prune

  – Do we really needs all the “small” leaves with perfect coverage?

Decision Tree Pruning Methodologies
-----

- Pre-pruning (top-down)

   – Stopping criteria while growing the tree

- Post-pruning (bottom-up)

   – Grow the tree, then prune –More popular
