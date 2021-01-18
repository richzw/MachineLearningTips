// https://math.stackexchange.com/questions/3421665/do-the-vertical-bar-and-semicolon-denote-the-same-thing-in-the-context-of-condit#:~:text=1%20Answer&text=Although%20some%20writers%20are%20a,similar%2C%20but%20not%20quite%20identical.

Q: 

  In the CMU Machine Learning Lecture, likelihood function is denoted by 𝑃(𝐷|𝜃)

  In the Cornell lecture note, likelihood function is denoted by 𝑃(𝐷;𝜃)

A:

- The `Sheffer stroke` (vertical bar) is read **"given that"** or **"contingent upon."** Thus 𝑝(𝑥|𝑦) means __the probability density of 𝑥 given that some event or state of affairs 𝑦 has occurred__. 
For instance if 𝑥 represents a height in centimeters, the term 𝑝(𝑥|woman) means the probability density we find the height value 𝑥 given that the person we are measuring is a woman.
This is classical terminology of `contingent probability density`. This use is restricted to probability and statistics.

- The semicolon is slightly different, and can apply in cases __unrelated to probability and statistics__. Suppose you have a mathematical function that depends upon two (or more) variables, 
say `𝑓(𝑥,𝑦)`. Suppose you're primarily interested in the behavior of 𝑓 based on the value of 𝑥. You want to take derivatives with respect to 𝑥, 
limits for large and small 𝑥, count the zeros, and such. When you write __𝑓(𝑥;𝑦) you're in essence creating a new function OF ONE VARIABLE (𝑥)__, 
which we might call `𝑔(𝑥)=𝑓(𝑥;𝑦)`. This notation recognizes that there is an implied or chosen value of 𝑦 when you examine 𝑔(𝑥), but it is not __"part of the function."__
For instance, it is meaningless to __take the derivative of 𝑓(𝑥;𝑦) with respect to 𝑦 any more than it would to take the derivative of 𝑔(𝑥) with respect to 𝑦. 
The value of 𝑦 is a "setting" or a "parameter" that is fixed.__

A particularly useful application of this semicolon notation in probability is in the `Expectation-Maximization (or EM)` Algorithm. 
In the algorithm some steps involve the optimization of a function (the expectation) with respect to one of the many variables. 
It is natural, then, to use `𝑓(𝑥;𝑦)` during this step to optimize over 𝑥, where the value of 𝑦 is fixed and __"cannot be touched."__

Another use in statistics and pattern recognition is the following. Suppose you have training data, `𝑥1,𝑥2,…,𝑥𝑑.` You create the dot product with some real-valued 
weight vector `𝐰={𝑤1,𝑤2,…,𝑤𝑑}`, as in `𝐰𝑡𝐱=𝑤1𝑥1+𝑤2𝑥2+…+𝑤𝑑𝑥𝑑`. You want to optimize the value of `𝐰` with respect to some classification. 
You create a function `ℎ(𝐰;𝐱)`, meaning you can take derivatives with respect to the weights but it makes NO SENSE to take a derivative with respect to the data 𝐱, 
nor is `𝐰 "contingent upon" 𝐱`. Nevertheless, the particular functional form of ℎ has buried behind it the particular values of the data at hand.

