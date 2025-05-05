#这里介绍一个autograd的包
#autograd包为Tensor上所有操作提供自动求导
#它是一个由运行定义的框架，这意味这以代码定义的方式定义反向传播，并且每次迭代都可以不同
#怎么理解这句话呢？

#首先：由运行定义的框架
#TensorFlow中需要提前定义好整个计算图，之后再运算
#而pytorch中在运行时动态构建计算图

#其次：以代码定义的方式定义反向传播
#在由运行定义的框架中，反向传播的过程是依靠前向传播时实际运行的代码来自动确定的
#这意味着只需要编写前向传播的代码，框架会根据这些代码自动记录每一步操作

#最后：每次迭代都可以不同
#由于计算图在运行时时动态构建的，所以每次迭代时，前向传播的代码逻辑可以不同，计算图也会随之改变。

#torch.Tensor是包的核心类
#如果将其属性.requires_grad设置为True，那么它将开始跟踪对于该张量的所有操作
#当完成计算后，可以调用.backward()来自动计算所有梯度
#这个张量的所有梯度将会自动累加到.grad属性中

#创建一个张量来追踪与它相关的计算
import torch
x=torch.ones(2,2,requires_grad=True)    #创建一个2x2的张量，并且设置requires_grad=True
print(x)
print("针对张量做一个操作")
y=x+2
#这里输出,grad_fn=<AddBackward0>,其中grad_fn表示梯度函数,它记录了这个张量是如何从其他张量计算而来的
#而AddBackward0表示这个张量是通过加法操作计算而来的,Backward0表示这个张量是通过反向传播计算而来的
print(y)
print(y.grad_fn)
print("针对y做一个操作")
z=y*y*3
out=z.mean()#.mean()是求平均值的函数
print(z,out)


#演示张量梯度运算
p=torch.ones(2,2,requires_grad=True)
q=p+2
r=q*q*3
out=r.mean()
#反向传播
out.backward()
#思考这里提到的“梯度”具体指的是什么？
print('p的梯度为：',p.grad)

