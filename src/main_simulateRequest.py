from calculateSimilar import calculateSimilarIterm4User,calculateSimilarUser4Iterm
import sys

usrNo = input("请输入用户编号（如'14700', '15462', '15318'）：")
print('用户编号：' + usrNo + '推荐的前10个物品为：')
print(calculateSimilarIterm4User(usrNo,num=10))

itermNo = input("请输入商品编号（如'21693', '84678', '85026B', '90210B'）：")
print('商品编号：' + itermNo + '推荐的前10个用户为：')
print(calculateSimilarUser4Iterm(itermNo,num=10))