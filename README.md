[[_TOC_]]

# 基于协同过滤的在线鲜花店推荐系统

<font color=#999AAA >

# 项目需求：

<font color=#999AAA >

* 基于店铺的客户订单记录，实现店铺的推荐需求：
  * 基于RFM模型，得到客户的价值分类，对高价值客户进行重点跟踪，推荐其潜在的商品列表，即实现：给定用户编号，返回10个推荐商品列表。
  * 对店铺滞销商品，进行有针对性的促销活动，推荐给最有可能购买的10个用户，结合一些针对性的促销优惠活动，向10个用户推荐。即实现：给定物品编号，返回10个推荐用户列表。

* 店铺尚未搭建Spark大数据环境，可搭建TensorFlow2的环境，因此使用TensorFlow2实现协同过滤算法中的矩阵分解。现有资料绝大部分是基于一个模板复制出来的，且基于TensorFlow1，因此需要亲自动手，用TensorFlow2实现。

* 若搭建好了Spark环境，可在Spark中，直接调用spark.mllib.recommendation.ALS() ，可实现相同功能。
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 推荐算法原理：

<font color=#999AAA >在推荐系统中，协同过滤算法是很常用的推荐算法。中心思想：物以类聚（以物品相似度推荐物品），人以群分（以用户相似度推荐类似用户的商品列表）。也就是口味相同的人，把他喜欢的物品或者电影歌曲推荐给你；或者是将你喜欢的物品，类似的物品推荐给你。

* 整体流程：
  1、 获取用户对商品的评分、购买记录等
  2、 构造协同矩阵M
  3、 基于矩阵进行分解M=U*V
  4、 利用要推荐的物品或者用户，和U或者V计算相似度

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 计算相似度：

<font color=#999AAA >TensorFlow2可以自动帮你求导更新参数，太方便了，你要做的就是构造损失函数loss而已。
loss函数可以理解为，我们分解得到U*V得到预测的M_pre，用M和M_pre求欧式距离：即欧几里得距离（Euclidean Distance）
公式具体为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2543ccfd3b85495ebfe5ac9bb10549f8.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5rWq5ryr55qE5pWw5o2u5YiG5p6Q,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

大致意思就是分解一个大矩阵为两个小矩阵相乘。
![在这里插入图片描述](https://img-blog.csdnimg.cn/447e4dde48ae43368127d33003b4717d.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5rWq5ryr55qE5pWw5o2u5YiG5p6Q,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

* 注意事项：

  * **欧氏距离能够体现个体数值特征的绝对差异，所以更多的用于需要从维度的数值大小中体现差异的分析，如使用用户行为指标分析用户价值的相似度或差异。**
  * **余弦距离更多的是从方向上区分差异，而对绝对的数值不敏感，更多的用于使用用户对内容评分来区分兴趣的相似度和差异，同时修正了用户间可能存在的度量标准不统一的问题（因为余弦距离对绝对数值不敏感）**。
  * **用户对内容评分，按5分制，X和Y两个用户对两个内容的评分分别为（1,2）和（4,5），使用余弦相似度得到的结果是0.98，两者极为相似。但从评分上看X似乎不喜欢2这个 内容，而Y则比较喜欢，余弦相似度对数值的不敏感导致了结果的误差，需要修正这种不合理性就出现了调整余弦相似度，即所有维度上的数值都减去一个均值，比如X和Y的评分均值都是3，那么调整后为（-2，-1）和（1,2），再用余弦相似度计算，得到-0.8，相似度为负值并且差异不小，但显然更加符合现实。**

* ***余弦相似度的python实现\***

  * python原生代码：

    ```python
    import numpy as np
    
    def cosine_similarity(x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)
    
        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
    
    
    if __name__ == '__main__':
        print cosine_similarity([0, 0], [0, 0])  # 1.0
        print cosine_similarity([1, 1], [0, 0])  # 0.0
        print cosine_similarity([1, 1], [-1, -1])  # -1.0
        print cosine_similarity([1, 1], [2, 2])  # 1.0
        print cosine_similarity([3, 3], [4, 4])  # 1.0
        print cosine_similarity([1, 2, 2, 1, 1, 1, 0], [1, 2, 2, 1, 1, 2, 1])  # 0.938194187433
        
    ```

    ![image-20211021205015569](C:\Users\luoji\AppData\Roaming\Typora\typora-user-images\image-20211021205015569.png)

# 矩阵分解TensorFlow2代码：

<font color=#999AAA >TensorFlow2可以自动帮你求导更新参数，太方便了，你要做的就是构造损失函数loss而已。

![image-20211021205200629](C:\Users\luoji\AppData\Roaming\Typora\typora-user-images\image-20211021205200629.png)

具体代码为：

```python
'''=================================================
@Function -> 用TensorFlow2实现协同过滤矩阵的分解
@Author ：luoji
@Date   ：2021-10-19
=================================================='''


def matrixDecomposition(alike_matrix,rank=10,num_epoch= 5000,learning_rate=0.001,reg=0.01):
    row,column = len(alike_matrix),len(alike_matrix[0])
    avg = np.average(alike_matrix)
    matrix_avged = alike_matrix-avg
    # 这里解释一下为什么要减去平均值进行归一化
    # 因为余弦相似度在数值上的不敏感，会导致这样一种情况存在：
    #
    # 用户对内容评分，按5分制，X和Y两个用户对A,B两个内容的评分分别为（1, 2）和（2, 4），使用余弦相似度得到的结果是1
    # ，两者极为相似。但从评分上看X似乎不喜欢B这个内容，而Y则比较喜欢，余弦相似度对数值的不敏感导致了结果的误差，需要修正这种不合理性就出现了调整余弦相似度，
    # 即所有维度上的数值都减去一个均值，比如X和Y的评分均值都是3，那么调整后为（-1.25，-0.25）和（0.25, 1.25），再用余弦相似度计算，得到 -0.38，相似度为负值并且差异不小，但显然更加符合现实。
    # 那么是否可以在（用户 - 商品 - 行为数值）矩阵的基础上进行调整,使用余弦相似度计算比普通余弦夹角算法要强。
    #
    # 欧氏距离能够体现个体数值特征的绝对差异，所以更多的用于需要从维度的数值大小中体现差异的分析，如使用用户行为指标分析用户价值的相似度或差异。

    y_true = tf.constant(matrix_avged, dtype=tf.float32)  # 构建y_true
    U = tf.Variable(shape=(row, rank), initial_value=np.random.random(size=(row, rank)),dtype=tf.float32)  # 构建一个变量U，代表user权重矩阵
    V = tf.Variable(shape=(rank, column), initial_value=np.random.random(size=(rank, column)),dtype=tf.float32)  # 构建一个变量，代表权重矩阵，初始化为0

    variables = [U,V]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_epoch):
        with tf.GradientTape() as tape:
          y_pre = tf.matmul(U, V)
          loss = tf.reduce_sum(tf.norm(y_true-y_pre, ord='euclidean')
                               + reg*(tf.norm(U,ord='euclidean')+tf.norm(V,ord='euclidean')))  #正则化项
          print("batch %d : loss %f" %(batch_index,loss.numpy()))

        grads = tape.gradient(loss,variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,variables))
    return U,V,tf.matmul(U, V)+avg

if __name__ == "__main__":
    # 把矩阵分解为 M=U*V ，U和V由用户指定秩rank
    alike_matrix = [[1.0, 2.0, 3.0],
                   [4.5, 5.0, 3.1],
                   [1.0, 2.0, 3.0],
                   [4.5, 5.0, 5.1],
                   [1.0, 2.0, 3.0]]
    U,V,preMatrix = matrixDecomposition(alike_matrix,rank=2,reg=0.5,num_epoch=2000) # reg 减小则num_epoch需增大

    print(U)
    print(V)
    print(alike_matrix)
    print(preMatrix)
    print("this difference between alike_matrix and preMatrix is :")
    print(alike_matrix-preMatrix)
    print('loss is :',sum(sum(abs(alike_matrix - preMatrix))))
```

待分解的矩阵：
[[1.0, 2.0, 3.0],
                   [4.5, 5.0, 3.1],
                   [1.0, 2.0, 3.0],
                   [4.5, 5.0, 5.1],
                   [1.0, 2.0, 3.0]]

分解后，相乘的到的矩阵：

[[1.0647349 1.929376  2.9957888]
 [4.6015587 4.7999315 3.1697667]
 [1.0643657 1.9290545 2.9957101]
 [4.287443  5.211667  4.996485 ]
 [1.0647217 1.9293401 2.9957187]],

可以看出两者还是很相似的，证明我们用TensorFlow2进行的矩阵分解是正确的。
注意，正则化项reg需要和num_epoch配套，reg越大，收敛越快，但效果不一定最好。
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 项目产出和亮点：

<font color=#999AAA >利用TensorFlow2，实现协同过滤算法中的矩阵分解，而且该模块可以直接复用。满足需求。

* 给定用户编号，返回10个推荐商品列表。
* 给定物品编号，返回10个推荐用户列表。

**亮点**：

* TensorFlow2太神奇了，只要找到损失函数loss，模型就可以训练。Amazing！
* 利用TensorFlow2进行的矩阵分解，目前是所有分解算法中误差最小的。
* 全网找不到第二家基于TensorFlow2实现的。在Spark中，直接调用spark.mllib.recommendation.ALS() 就好了。

# 代码地址github：

<font color=#999AAA >

* 源码地址：

  []: https://github.com/jiluojiluo/recommenderSystemForFlowerShop	"GitHub"

  

* CSDN 技术博客地址：

  [https://blog.csdn.net/weixin_43290383/article/details/120895031]()

项目版本号：

* tf.__version__
  '2.5.0'
* Python 3.8.8
* java version "1.8.0_121"

1、加深了对TensorFlow2的理解，太神奇了，只要找到损失函数loss，模型就可以训练。Amazing！
2、CSDN 技术博客1 篇，全网找不到第二个基于TensorFlow2实现的。好奇为什么TensorFlow2不帮我们实现了，在Spark中，直接调用spark.mllib.recommendation.ALS() 就好了