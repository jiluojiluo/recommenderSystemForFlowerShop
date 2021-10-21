'''=================================================
@Function -> 用TensorFlow2实现协同过滤矩阵的分解
@Author ：luoji
@Date   ：2021-10-19
=================================================='''

import numpy as np
import tensorflow as tf

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
    # alike_matrix = [[1.0, 2.0, 3.0,4.0],
    #                [5.0, 4.0, 3.1,1.0],
    #                [1.0, 2.0, 0.0,0.0],
    #                [1.0, 2.0, 3.0,4.0],
    #                [1.0, 2.0, 3.0,5.0]]
    # U,V,preMatrix = matrixDecomposition(alike_matrix,rank=2,reg=0.01,num_epoch=2000) # reg 减小则num_epoch需增大
    #
    # print(U)
    # print(V)
    # print(alike_matrix)
    # print(preMatrix)
    # print("this difference between alike_matrix and preMatrix is :")
    # print(alike_matrix-preMatrix)
    # print('loss is :',sum(sum(abs(alike_matrix - preMatrix))))


    alike_matrix =  [[5,3,0,1],
                     [4,0,0,1],
                     [1,1,0,5],
                     [1,0,0,4],
                     [0,1,5,4]]
    U, V, preMatrix = matrixDecomposition(alike_matrix, rank=2, reg=0.02, num_epoch=10000,learning_rate=0.0002)  # reg 减小则num_epoch需增大

    print(U)
    print(V)
    print(alike_matrix)
    print(preMatrix)
    print("this difference between alike_matrix and preMatrix is :")
    print(alike_matrix - preMatrix)
    print('loss is :', sum(sum(abs(alike_matrix - preMatrix))))
    # loss is : tf.Tensor(10.604275, shape=(), dtype=float32)
#
#
# [[-0.05648386  0.0709362  -0.02550554]
#  [-0.13646317  0.17138195 -0.06162167]
#  [-0.05648398  0.0709362  -0.0255053 ]
#  [ 0.18826008 -0.2364316   0.08501101]
#  [-0.0564841   0.0709362  -0.02550507]], shape=(5, 3), dtype=float32)
# loss is : tf.Tensor(1.3379459, shape=(), dtype=float32)                              reg=0
#
# tf.Tensor(
# [[-0.07570493  0.08035505  0.03565478]
#  [-0.0538125   0.22750902 -0.07667994]
#  [-0.07686806  0.07968223  0.03636932]
#  [ 0.23263788 -0.18698311  0.12599993]
#  [-0.07711768  0.07959962  0.03669596]], shape=(5, 3), dtype=float32)
# loss is : tf.Tensor(1.48167, shape=(), dtype=float32)                                 reg=1
#
#
# tf.Tensor(
# [[-0.06473494  0.07062399  0.00421119]
#  [-0.10155869  0.20006847 -0.06976676]
#  [-0.06436574  0.0709455   0.00428987]
#  [ 0.21255684 -0.21166706  0.10351467]
#  [-0.0647217   0.07065988  0.00428128]], shape=(5, 3), dtype=float32)
# loss is : tf.Tensor(1.3179666, shape=(), dtype=float32)                               reg=0.5
#
#
# tf.Tensor(
# [[-0.05662596  0.07102382 -0.02494812]
#  [-0.13573742  0.17191172 -0.06174588]
#  [-0.05662465  0.07102513 -0.0249474 ]
#  [ 0.18865538 -0.23597574  0.08540916]
#  [-0.05662537  0.0710243  -0.02494836]], shape=(5, 3), dtype=float32)
# loss is : tf.Tensor(1.3372284, shape=(), dtype=float32)                                reg=0.1



