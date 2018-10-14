Definition
-----

Consider the error of hypothesis ℎ. We let error on the training data be error𝑡𝑟𝑎𝑖𝑛 ℎ and error over the entire distribution 𝐷 of data be error𝐷 ℎ .

Then a hypothesis ℎ “overfits” the training data if there is an alternative hypothesis, ℎ′, such that:

    error𝑡𝑟𝑎𝑖𝑛(ℎ) < error𝑡𝑟𝑎𝑖𝑛(ℎ′)
    error𝐷(ℎ) < error𝐷(ℎ′)

Errors committed by classification models are generally divided into two types:

- **Training Errors**

    The number of misclassification errors committed on training records; also called resubstitution error.
- **Generalization Errors**

    The expected error of the model on previously unseen records.

Causes of Overfitting
-------

- **Overfitting Due to Presence of Noise**

    Mislabeled instances may contradict the class labels of other similar records.

- **Overfitting Due to Lack of Representative Instances**

    Lack of representative instances in the training data can prevent refinement of the learning algorithm.

- **Overfitting and the Multiple Comparison Procedure**

    Failure to compensate for algorithms that explore a large number of alternatives can result in spurious fitting


