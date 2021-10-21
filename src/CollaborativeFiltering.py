import pandas as pd
import numpy as np
import MatrixDecomposition

tradeData = pd.read_csv('C:\LuojiPythonProject\YiSuXianHuaTuiJian\一束鲜花订单记录.csv',dtype='str')
print(tradeData.shape)

iterm = tradeData['产品码'].drop_duplicates().reset_index()
iterm_size = len(iterm)
user = tradeData['用户码'].drop_duplicates().reset_index()
user_size = len(user)

consume_times = tradeData[{'用户码','产品码'}]

# 为物品编号Id，从0开始，物品编码可能有字母等。
iterm['itermId']=iterm.index
iterm=iterm.drop(columns=['index'])
# 保存物品编号Id和物品码的对应关系，方便以后查找具体物品。
pd.DataFrame.to_csv(iterm,"C:\LuojiPythonProject\YiSuXianHuaTuiJian\iterm2Id.csv",columns=['产品码','itermId'],encoding="GBK")
index2ItermDict=dict( zip(iterm['itermId'],iterm['产品码']))
iterm2IndexDict=dict( zip(iterm['产品码'],iterm['itermId']))
# 保存物品编号ID和物品码的对应关系，方便以后查找具体物品。

# 为用户编号，从0开始，用户编码可能有字母等。
user['userId']=user.index
user=user.drop(columns=['index'])
# 保存用户编号Id和用户码的对应关系，方便以后查找具体物品。
pd.DataFrame.to_csv(user,"C:\LuojiPythonProject\YiSuXianHuaTuiJian\myUser2Id.csv",columns=['用户码','userId'],encoding="GBK")
user2IndexDict=dict(zip(user['用户码'],user['userId']))
index2UserDict=dict(zip(user['userId'],user['用户码']))

# 增加一列计数列，从1开始
consume_times['times']=int(1)
# 把产品码和用户码替换成id，便于构造相关矩阵。
consume_times['产品码']=[iterm2IndexDict[x] for x in consume_times['产品码']]
consume_times['用户码']=[user2IndexDict[x] for x in consume_times['用户码']]

# 统计消费者购买商品的次数。这里仅用到了次数特征。可以按照自己的想法构造特征，比如金额信息，或者RFM模型中生成的因子。只要是类似电影评分那样的数值型就OK
secoreMatrixDatafreme = pd.DataFrame(consume_times.groupby(by=["用户码","产品码"]).agg('count'))
# 变换矩阵的索引列
x=secoreMatrixDatafreme.index.to_frame()
x.rename(columns={'用户码':'u','产品码':'i'},inplace = True)
# 把统计到的购买次数和产品码用户码拼接起来
# TODO 这里可以根据实际业务进行评分矩阵打分的构造，比如加入购买总数量和购买金额，注意考虑个维度的取值范围，也可以先进行归一化各个特征，然后按权重加总
scoreFeture= x.merge(secoreMatrixDatafreme, on=['用户码', '产品码'])
scoreFeture = scoreFeture.reset_index(drop=True)

# 得到评分矩阵
secoreMatrixArray = np.array(scoreFeture)
ratings_df =scoreFeture
userNo = ratings_df['u'].max() + 1     #总得用户数
itermNo = ratings_df['i'].max() + 1    #总得物品数
rating = np.zeros((userNo,itermNo))

#标志位
flag = 0
#获取合并表中的列数
ratings_df_length = np.shape(ratings_df)[0]
#遍历矩阵，将电影的评分（物品购买次数）填入表中
for index,row in ratings_df.iterrows():
    rating[int(row['u']), int(row['i'])] = row['times']
    flag += 1
    # print('processed %d, %d left' %(flag,ratings_df_length-flag))

record = rating > 0
record = np.array(record, dtype = int)
print(record)  #表示有关联关系的矩阵，没用
print(rating)

U,V,pre_Maxtirx = MatrixDecomposition.matrixDecomposition(alike_matrix=rating,rank=10,learning_rate=0.0005,num_epoch=12000,reg=0.1)
print("this difference between alike_matrix and preMatrix is :")
print(rating - pre_Maxtirx)
print('loss is :', sum(sum(abs(rating - pre_Maxtirx))))
U = pd.DataFrame(U.numpy())
pd.DataFrame(U).to_csv("C:\LuojiPythonProject\YiSuXianHuaTuiJian\EmbedingUser.csv")
VT=np.array(V).transpose()
VT=pd.DataFrame(VT)
pd.DataFrame(VT).to_csv("C:\LuojiPythonProject\YiSuXianHuaTuiJian\EmbedingIterm.csv")

print("矩阵分解得到物品和用户的embeding向量结束")