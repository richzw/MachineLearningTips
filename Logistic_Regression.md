
极大似然估计是概率论在统计学中的应用。它提供了一种给定观察数据来评估模型参数的方法，即：“模型已定，参数未知”。
通过若干次试验，观察其结果，利用试验结果得到某个参数值能够使样本出现的概率为最大，则称为极大似然估计

在线性回归中，我们采取了一种计算方式叫做最小二乘法，但是这个方法并不适用于逻辑回归。
sigmoid的加入让我们的损失函数变成一种非凸函数，中间有非常多的局部最小值，简单来说，就是不能像坐滑梯一样顺利的滑到全局最低点，而是会中途掉进某个坑里。

Source: https://blog.csdn.net/qq_26413541/article/details/84113135
