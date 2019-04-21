
- Question:

  [Use to k-fold cross validation to select better Decision tree model?](https://stackoverflow.com/questions/2314850/help-understanding-cross-validation-and-decision-trees)
  
  **The purpose of cross validation is not to help select a particular instance of the classifier** (or decision tree, or whatever automatic learning application) but rather to qualify the model, i.e. to provide metrics such as the average error ratio, the deviation relative to this average etc. which can be useful in asserting the level of precision one can expect from the application. One of the things cross validation can help assert is whether the training data is big enough.
  
  we usually use the entire dataset for building the final model, but we use **cross-validation** (CV) to get a better estimate of the _generalization error_ on new unseen data.

  **With regards to selecting a particular tree**, you should instead run yet another training on 100% of the training data available, as this typically will produce a better tree. (The downside of the Cross Validation approach is that we need to divide the [typically little] amount of training data into "folds" and as you hint in the question this can lead to trees which are either overfit or underfit for particular data instances).

[Cross-validation methods](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f):

- Holdout Method
   it still suffers from issues of high variance
- K-fold 
   This significantly reduces bias as we are using most of the data for fitting, and also significantly reduces variance as most of the data is also being used in validation set
- Stratified K-Fold
   a slight variation in the K Fold cross validation technique is made, such that each fold contains approximately the same percentage of samples of each target class as the complete set, or in case of prediction problems, the mean response value is approximately equal in all the folds
- Leave-P-Out
   if there are n data points in the original sample then, n-p samples are used to train the model and p points are used as the validation set. This is repeated for all combinations in which original sample can be separated this way, and then the error is averaged for all trials, to give overall effectiveness.

[Bias and variance in leave-one-out vs K-fold cross validation](https://stats.stackexchange.com/questions/61783/bias-and-variance-in-leave-one-out-vs-k-fold-cross-validation/357749#357749)

(Examples)[https://www.ritchieng.com/machine-learning-cross-validation/]

- Cross-validation example: parameter tuning
- Cross-validation example: model selection 
- Cross-validation example: feature selection

