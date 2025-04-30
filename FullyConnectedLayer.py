#首先，全连接层是指一个多个神经元组成的神经网络层，其所有的输出和该层输入都有连接——>每个输入都会影响所有神经元的输出
#torch.nn.Linear()是一个用于构建线性层(全连接层)的类，它的功能是实现线性变换的操作公式为(y=xA^T+b)
#其中，x是输入张量，A是权重矩阵，b是偏置向量，y是输出张量

#基础用法
import torch.nn as nn
import torch

#torch.nn.Linear(in_features,out_features,bias=True,device=None,dtype=None)
#in_features:输入特征的维度(比如像素)
#out_features:输出特征的维度

#定义输入(batch_size=1,特征数=3)，批大小表示同时处理一个样本
input1=torch.tensor([[10.,20.,30.]])
#定义网络
linear_layer=nn.Linear(3,5)
#定义权重(这里的.data表示直接访问参数张量的方法(绕过梯度计算))
linear_layer.weight.data=torch.tensor([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.],[4.,4.,4.],[5.,5.,5.]])
#矩阵乘法(1,3)x(3,5)=(1,5)
#定义偏置
linear_layer.bias.data=torch.tensor(0.6)
#输出并打印输出
output=linear_layer(input1)
print(input1)
print(output,output.shape)