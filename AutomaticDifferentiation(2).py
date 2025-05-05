import torch
#雅克比向量积的梯度计算
#雅可比向量积的本质：函数在某个点x沿着方向v的变化率(J(x)*v)

x=torch.randn(3,requires_grad=True)
y=x*2
#这里的.norm()是pytorch中的一个函数，用来计算向量或者矩阵的范数(norm)
#默认情况下，.norm()函数计算的是欧几里得范数(根号下所有元素的平方和)
#这里的while循环是为了保证y的范数大于1000，是为了放大输出函数y与输入函数x的变化率
while y.data.norm()<1000:
    y=y*2

print('查看张量y')
print(y)
v=torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)
print('查看张量x的梯度')
print(x.grad)
#这里输出
#tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
#这里e+02代表10的2次方，e-01代表10的-1次方

#使用with torch.no_grad()停止跟踪求导
