import torch
import matplotlib.pyplot as plt

#首先得定义输入
#这里有linspace的用法
x=torch.linspace(-5,5,200)#从-5到5之间生成200个点
X=x.numpy()
#定义不同的激活函数
#y_relu=torch.relu(x).data.numpy()，这里已经不常会了，都是使用.dertach()去创建一个张量（不参与反向传播）
y_relu=torch.relu(x).detach().numpy()
y_sigmoid=torch.sigmoid(x).detach().numpy()
y_tanh=torch.tanh(x).detach().numpy()
#画图显示效果
#figure(编号，大小(单位是英寸))
plt.figure(1,figsize=(8,6))
#plt.subplot(行，列，位置(这里的位置表示第几幅图))
plt.subplot(221)
plt.plot(X,y_relu,c='red',label='relu')
#plt.legend()显示图例
#基本用法：在plot的时候，给每条线加上label,然后用legend()显示出来
plt.legend(loc='best')
plt.subplot(222)
plt.plot(X,y_sigmoid,c='blue',label='sigmoid')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(X,y_tanh,c='green',label='tanh')
plt.legend(loc='best')
plt.show()
