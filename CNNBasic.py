#这里遇到最需要思考的问题,为什么需要使用到转置，直接在设计网络层面的时候反过来不久行了吗？
import numpy as np
#N是批次大小(一次性处理样本的数量)，D_in是输入维度(特征数)，H是隐藏层维度(学习到的特征)，D_out是输出维度(结果特征数)
N,D_in,H,D_out=64,1000,100,10
#创建随机输入和输出数据
#x是shape为(N,D_in)的numpy数组，y是shape为(N,D_out)的numpy数组
x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)
#随机初始化权重
w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)
learning_rate=1e-6
#循环500次
for t in range(500):
    #前向传播：计算预测值y
    #输入与第一层权重相乘，得到隐藏层的输入
    h=x.dot(w1)
    #把小于0的数置为0，得到隐藏层的输出
    h_relu=np.maximum(h,0)
    #隐藏层的输出与第二层权重相乘，得到输出层的输入
    y_pred=h_relu.dot(w2)
    #计算和打印损失
    #使用均方误差：每个输出元素平方差再求和
    loss=np.square(y_pred-y).sum()
    if t%100==0:
        print(t,loss)
    #反向传播，计算w1和w2对损失的梯度
    grad_y_pred=2.0*(y_pred-y)#对y_pred求导
    grad_w2=h_relu.T.dot(grad_y_pred)#.T表示转置，.dot表示矩阵乘法
    grad_h_relu=grad_y_pred.dot(w2.T)#对h_relu求梯度
    grad_h=grad_h_relu.copy()
    grad_h[h<0]=0#由于Reluy的倒数是0或1，所以这里把小于0的数置为0
    grad_w1=x.T.dot(grad_h)#对w1求梯度

    #求导vs求梯度
    #导数是函数在某一点的变化率，而梯度是函数在某一点的变化率的向量

    #更新权重
    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2