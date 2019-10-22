
Loss Func 损失函数概览
----

https://zhuanlan.zhihu.com/p/38529433

优化器算法详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）
----

对于优化算法，优化的目标是网络模型中的参数θ（是一个集合，θ1、θ2、θ3 ......）目标函数为损失函数L = 1/N ∑ Li （每个样本损失函数的叠加求均值）。这个损失函数L变量就是θ，其中L中的参数是整个训练集，换句话说，目标函数（损失函数）是通过整个训练集来确定的，训练集全集不同，则损失函数的图像也不同。那么为何在mini-batch中如果遇到鞍点/局部最小值点就无法进行优化了呢？因为在这些点上，L对于θ的梯度为零，换句话说，对θ每个分量求偏导数，带入训练集全集，导数为零。对于SGD/MBGD而言，每次使用的损失函数只是通过这一个小批量的数据确定的，其函数图像与真实全集损失函数有所不同，所以其求解的梯度也含有一定的随机性，在鞍点或者局部最小值点的时候，震荡跳动，因为在此点处，如果是训练集全集带入即BGD，则优化会停止不动，如果是mini-batch或者SGD，每次找到的梯度都是不同的，就会发生震荡，来回跳动。

https://www.cnblogs.com/guoyaohua/p/8542554.html