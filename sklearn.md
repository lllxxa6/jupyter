## 1.1 线性模型
下面是一组用于回归的方法，其中目标值是特征的线性组合。在数学表示法中如果$\widehat{y}$表示预测值，那么有：
$$\widehat{y}(w,x) = w_0+w_1x_1+...+w_px_p$$
在整个模型中，我们定义向量$w=(w_1,...,w_p)$作为`coef_`，定义$w_0$作为`intercept_`
若需要使用广义线性模型进行分类， 请看 [LinearRegression](https://scikit-learn.org.cn/view/394.html)
### 1.1.1 普通最小二乘法
[LinearRegression](https://scikit-learn.org.cn/view/394.html)是拟合一个带有回归系数的$w=(w_1,...,w_p)$，使得数据的实际观测值和线性近似预测的预测值之间的残差平方和最小的线性模型。数学上讲，他解决了这个形式的问题：
$$\underset{w}{min}||Xw-y||_2^2$$

<div align=center><img src="https://scikit-learn.org.cn/upload/739a16781ef534c5ae4cf93a5f700082.png" ></div>

[LinearRegression](https://scikit-learn.org.cn/view/394.html)将采用它的`fit`方法去拟合数组$X，y$，并将线性模型的回归系数$w$存储在`coef_`中：
```python
>>> from sklearn import linear_model
>>> reg = linear_model.LinearRegression()
>>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression()
>>> reg.coef_
array([0.5, 0.5])
```
普通最小二乘的系数估计依赖于特征的独立性。当特征相关且设计矩阵的列之间具有近似线性相关性时， 设计矩阵趋于奇异矩阵，最小二乘估计对观测目标的随机误差高度敏感，可能产生很大的方差。例如，在没有实验设计的情况下收集数据时，就可能会出现这种多重共线性的情况。

|                           示例                           |
| :------------------------------------------------------: |
| [线性回归实例](http://scikit-learn.org.cn/view/230.html) |
#### 1.1.1.1 普通最小二乘法的复杂度
利用$X$的奇异值分解计算最小二乘，如果$X$的形状是`(n_samples,n_features)`，假设$n\_samples \ge n\_features$这个算法的复杂度是$O(n\_samples\ n\_features^2)$。
### 1.1.2 岭回归与分类
#### 1.1.2.1 回归
Ridge 通过对系数的大小施加惩罚来解决普通最小二乘的一些问题。岭系数最小化一个带惩罚项的残差平方和:
$$\underset{w}{min}||Xw-y||_2^2+\alpha|||w|||_2^2$$
其中，$\alpha \ge 0$是控制收缩量的复杂性参数，$\alpha$值越大，收缩量越大，这样，系数对共线性的鲁棒性就更强了。

<div align=center><img src="https://scikit-learn.org.cn/upload/ee4e3a4a8d65422dd03361b47b43c9b8.png" ></div>

与其他线性模型一样，Ridge 用`fit`方法完成拟合，并将模型系数$w$存储在其`coef_`成员中：
```python
>>> from sklearn import linear_model
>>> reg = linear_model.Ridge(alpha=.5)
>>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
Ridge(alpha=0.5)
>>> reg.coef_
array([0.34545455, 0.34545455])
>>> reg.intercept_
0.13636...
```
#### 1.1.2.2 分类
Ridge有一个分类器变体：RidgeClassifier。该分类器首先将二分类目标转换为`{-1,1}`，然后将问题作为一个回归任务处理，以优化上述目标。预测类对应于回归预测的符号。对于多分类，该问题被视为多输出回归，预测的类对应于具有最高值的输出。

使用(受惩罚的)最小二乘损失来拟合分类模型，而不是更传统的 logistic损失或hinge损失，似乎有些问题。然而，在实践中，所有这些模型都可能导致类似的交叉验证分数， 例如准确性或精确度/召回率，而受惩罚的最小二乘损失使用的岭分类有一个明显的计算机性能剖面选择。

RidgeClassifier可以明显快于LogisticRegression，因为它只计算投影矩阵一次。

这种分类器有时被称为具有线性核的最小二乘支持向量机。

| 示例                                                                                                                                                                                                                   |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [正则化函数的岭系数图](http://scikit-learn.org.cn/view/219.html)<br>[基于稀疏特征的文本文档分类](http://scikit-learn.org.cn/view/354.html)<br>[线性模型系数解释中的常见陷阱](http://scikit-learn.org.cn/view/257.html) |
#### 1.1.2.3 岭复杂度
该方法的复杂度与[普通最小二乘法](#1111-普通最小二乘法的复杂度)相同。
#### 1.1.2.4 设置正则化参数：广义交叉验证
RidgeCV通过设置的alpha参数的控制交叉验证来实现岭回归。该对象与GridSearchCV的使用方法相同，只是它默认为Generalized Cross-Validation(广义交叉验证 GCV)，这是一种有效的留一交叉验证方法（LOO-CV）:
```python
>>> import numpy as np
>>> from sklearn import linear_model
>>> reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
>>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
RidgeCV(alphas=array([1.e-06, 1.e-05, 1.e-04, 1.e-03,1.e-02, 1.e-01, 1.e+00, 1.e+01,1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]))
>>> reg.alpha_
0.01
```
指定cv属性的值将触发使用GridSearchCV进行交叉验证，例如，如`cv=10`执行10折交叉验证，而不是广义交叉验证。
> 参考
>“Notes on Regularized Least Squares”, Rifkin & Lippert (technical report, course slides).

### 1.1.3 Lasso
Lasso 是一个估计稀疏系数的线性模型。它在某些情况下是有用的，因为它倾向于给出非零系数较少的解，从而有效地减少了给定解所依赖的特征数。因此，Lasso 及其变体是压缩感知领域的基础。在一定条件下，可以恢复非零系数的精确集合。[见压缩感知：用L1优先层重建(Lasso))](http://scikit-learn.org.cn/view/185.html)。
从数学上讲，它由一个带正则项的线性模型组成。最小化的目标函数：

$$\underset{w}{min}\frac{1}{2n_{samples}}||Xw-y||_2^2+\alpha||w||_1$$

这样，LASSO估计器解决了最小二乘损失加惩罚项$\alpha||w||_1$的最优化问题，其中$\alpha$是常数，$||w||_1$是系数向量的$l1$范数。

在Lasso类中的实现采用坐标下降法作为拟合系数的算法。另一个算法 见最小角回归 。

```python
>>> from sklearn import linear_model
>>> reg = linear_model.Lasso(alpha=0.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])
Lasso(alpha=0.1)
>>> reg.predict([[1, 1]])
array([0.8])
```
函数lasso path对于较低级别的任务很有用，因为它计算沿着可能值的全部路径上的系数。

| 示例                                                                                                                                                                                                                                                  |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Lasso和Elastic Net(弹性网络)在稀疏信号上的表现](http://scikit-learn.org.cn/view/240.html)<br>[压缩感知：L1先验层析重建(Lasso)](http://scikit-learn.org.cn/view/185.html)<br>[线性模型系数解释中的常见陷阱](http://scikit-learn.org.cn/view/257.html) |
**注意: 使用 Lasso 进行特征选择**

由于Lasso回归产生稀疏模型，因此可以用来进行特征选择。详见：基于 L1 的特征选取。

下面两篇参考解释了scikit-learn坐标下降算法中使用的迭代，以及用于收敛控制的对偶间隔计算的理论基础。

> 参考
> - “Regularization Path For Generalized linear Models by Coordinate Descent”, Friedman, Hastie & Tibshirani, J Stat Softw, 2010 (Paper).
> - “An Interior-Point Method for Large-Scale L1-Regularized Least Squares,” S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky, in IEEE Journal of Selected Topics in Signal Processing, 2007 (Paper)

#### 1.1.3.1 设置正则化参数
`alpha` 参数控制估计系数的稀疏程度。
##### 1.1.3.1.1 使用交叉验证
scikit-learn 设置Lasso`alpha`参数是通过两个公开的对象，[LassoCV](https://scikit-learn.org.cn/view/414.html)和[LassoLarsCV](https://scikit-learn.org.cn/view/417.html)。其中，[LassoLarsCV](https://scikit-learn.org.cn/view/417.html)是基于下面将要提到的[最小角回归]()算法。

对于具有多个共线特征的高维数据集， [LassoCV](https://scikit-learn.org.cn/view/414.html)通常是更可取的。然而，[LassoLarsCV](https://scikit-learn.org.cn/view/417.html)具有探索更多相关$\alpha$参数值的优点，如果样本数相对于特征数很少，则往往比[LassoCV](https://scikit-learn.org.cn/view/414.html)更快。

![](https://scikit-learn.org.cn/upload/2c63fc4f137bcafb40ce508de562e7a3.png)

##### 1.1.3.1.2 基于信息的模型选择
有多种选择时，估计器 [LassoLarsIC](https://scikit-learn.org.cn/view/418.html) 建议使用 Akaike information criterion （Akaike 信息判据）（AIC）或 Bayes Information criterion （贝叶斯信息判据）（BIC）。 当使用 k-fold 交叉验证时，正则化路径只计算一次而不是 k + 1 次，所以找到 α 的最优值是一种计算上更经济的替代方法。 然而，这样的判断需要对解决方案的自由度进行适当的估计，它会假设模型是正确的，对大样本（渐近结果）进行导出，即数据实际上是由该模型生成的。 当问题严重受限（比样本更多的特征）时，它们也容易崩溃。

<div align=center><img src="https://scikit-learn.org.cn/upload/158133e7bca26e63e7de5bb04970e17d.png" ></div>

| 示例                                                                       |
| :------------------------------------------------------------------------- |
| [Lasso模型选择-交叉验证/AIC/BIC](http://scikit-learn.org.cn/view/249.html) |

###### 1.1.3.1.3 与支持向量机正则化参数的比较
`alpha` 和 SVM 的正则化参数`C` 之间的等式关系是 `alpha = 1 / C` 或者 `alpha = 1 / (n_samples * C)`，并依赖于估计器和模型优化的确切的目标函数。

### 1.1.4 多任务Lasso
