
[OneHotEncoder vs DummyEncoder](https://stats.stackexchange.com/questions/224051/one-hot-vs-dummy-encoding-in-scikit-learn)

Scikit-learn's linear regression model allows users to disable intercept. So for one-hot encoding, should I always set fit_intercept=False? For dummy encoding, fit_intercept should always be set to True? I do not see any "warning" on the website.

For an unregularized linear model with one-hot encoding, yes, you need to set the intercept to be false or else incur perfect collinearity.  sklearn also allows for a ridge shrinkage penalty, and in that case it is not necessary, and in fact you should include both the intercept and all the levels. For dummy encoding you should include an intercept, unless you have standardized all your variables, in which case the intercept is zero.

Since one-hot encoding generates more variables, does it have more degree of freedom than dummy encoding?

The intercept is an additional degree of freedom, so in a well specified model it all equals out.

For the second one, what if there are k categorical variables? k variables are removed in dummy encoding. Is the degree of freedom still the same?

You could not fit a model in which you used all the levels of both categorical variables, intercept or not. For, as soon as you have one-hot-encoded all the levels in one variable in the model, say with binary variables x1,x2,…,xn, then you have a linear combination of predictors equal to the constant vector

`x1+x2+⋯+xn=1`

If you then try to enter all the levels of another categorical x′ into the model, you end up with a distinct linear combination equal to a constant vector

`x′1+x′2+⋯+x′k=1`

and so you have created a linear dependency

`x1+x2+⋯xn−x′1−x′2−⋯−x′k=0`

So you must leave out a level in the second variable, and everything lines up properly.

Say, I have 3 categorical variables, each of which has 4 levels. In dummy encoding, 3*4-3=9 variables are built with one intercept. In one-hot encoding, 3*4=12 variables are built without an intercept. Am I correct?

The second thing does not actually work. The 3×4=12 column design matrix you create will be singular. You need to remove three columns, one from each of three distinct categorical encodings, to recover non-singularity of your design.

