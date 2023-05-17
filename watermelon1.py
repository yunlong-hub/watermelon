# -*- coding: utf-8 -*-
#单隐层网络
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
seed = 2023
import random
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.close('all')

def preprocess():
    # 读取数据
    data = pd.read_table('watermelon30.txt',delimiter=',')
    data.drop('编号',axis=1,inplace=True)
    #1.将非数映射数字
    for title in data.columns:
        if data[title].dtype == 'object':
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
    #2.将数据顺数打乱，提升泛化
    data = np.array(data)
    np.random.shuffle(data)
    #3.数据的70%作为训练集和30%作为测试集
    n_samples = data.shape[0]  # 样本数量
    train_ratio = 0.7  # 训练集比例
    train_samples = int(train_ratio * n_samples)  # 训练集样本数量
    train_X = data[:train_samples,:-1]
    train_Y = data[:train_samples,-1]
    test_X = data[:train_samples,:-1]
    test_Y = data[train_samples:,-1]
    #4.去均值和方差归一化
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    train_x, train_y = np.array(train_X), np.array(train_Y).reshape(train_Y.shape[0],1)

    test_X = ss.fit_transform(test_X)
    test_x, test_y = np.array(test_X),np.array(test_Y).reshape(test_Y.shape[0],1)
    print(train_x, train_y, test_x, test_y)
    return train_x, train_y, test_x, test_y
#定义Sigmoid,求导
def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    return x*(1-x)

##累积BP算法
def accumulate_BP(x,y,dim=10,eta=0.1,max_iter=500):
    n_samples = x.shape[0]
    w1 = np.zeros((x.shape[1],dim))
    b1 = np.zeros((n_samples,dim))
    w2 = np.zeros((dim,1))
    b2 = np.zeros((n_samples,1))
    losslist = []
    for ite in range(max_iter):
        ##前向传播
        in1 = np.dot(x,w1)+b1
        out1 = sigmoid(in1)
        in2 = np.dot(out1,w2)+b2
        out2 = sigmoid(in2)
        loss = np.mean(np.square(y - out2))/2
        losslist.append(loss)
        print('iter:%d  loss:%.4f'%(ite,loss))
        ##反向传播
        ##标准BP
        d_out2 = out2 - y
        d_in2 = d_out2*d_sigmoid(out2)
        d_w2 = np.dot(np.transpose(out1),d_in2)
        d_b2 = d_in2
        d_out1 = np.dot(d_in2,np.transpose(w2))
        d_in1 = d_out1*d_sigmoid(out1)
        d_w1 = np.dot(np.transpose(x),d_in1)
        d_b1 = d_in1
        ##更新
        w1 = w1 - eta*d_w1
        w2 = w2 - eta*d_w2
        b1 = b1 - eta*d_b1
        b2 = b2 - eta*d_b2
    ##Loss可视化
    plt.figure()
    plt.plot([i+1 for i in range(max_iter)],losslist)
    plt.legend(['accumlated BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w1,w2,b1,b2

##标准BP算法
def standard_BP(x,y,dim=10,eta=0.1,max_iter=500):
    n_samples = 1
    w1 = np.zeros((x.shape[1],dim))
    b1 = np.zeros((n_samples,dim))
    w2 = np.zeros((dim,1))
    b2 = np.zeros((n_samples,1))
    losslist = []
    for ite in range(max_iter):
        loss_per_ite = []
        for m in range(x.shape[0]):
            xi,yi = x[m,:],y[m,:]
            xi,yi = xi.reshape(1,xi.shape[0]),yi.reshape(1,yi.shape[0])
            ##前向传播
            in1 = np.dot(xi,w1)+b1
            out1 = sigmoid(in1)
            in2 = np.dot(out1,w2)+b2
            out2 = sigmoid(in2)
            loss = np.square(yi - out2)/2
            loss_per_ite.append(loss)
            print('iter:%d  loss:%.4f'%(ite,loss))
            ##反向传播
            ##标准BP
            d_out2 = -(yi - out2)
            d_in2 = d_out2*d_sigmoid(out2)
            d_w2 = np.dot(np.transpose(out1),d_in2)
            d_b2 = d_in2
            d_out1 = np.dot(d_in2,np.transpose(w2))
            d_in1 = d_out1*d_sigmoid(out1)
            d_w1 = np.dot(np.transpose(xi),d_in1)
            d_b1 = d_in1
            ##更新
            w1 = w1 - eta*d_w1
            w2 = w2 - eta*d_w2
            b1 = b1 - eta*d_b1
            b2 = b2 - eta*d_b2
        losslist.append(np.mean(loss_per_ite))
    ##Loss可视化
    plt.figure()
    plt.plot([i+1 for i in range(max_iter)],losslist)
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w1,w2,b1,b2

def main():
    train_x, train_y, test_x, test_y = preprocess()
    #训练
    dim = 10
    w1,w2,b1,b2= standard_BP(x,y,dim)
    #w1,w2,b1,b2 = accumulate_BP(train_x, train_y,dim)
    #测试
    in1 = np.dot(test_x,w1)+b1
    out1 = sigmoid(in1)
    in2 = np.dot(out1,w2)+b2
    out2 = sigmoid(in2)
    y_pred = np.round(out2)
    print(y_pred,"/n")
    print(test_y)
    result = pd.DataFrame(np.hstack((test_y,y_pred)),columns=['真值','预测'] )
    result.to_excel('result_numpy.xlsx',index=False)

if __name__=='__main__':
    main()

