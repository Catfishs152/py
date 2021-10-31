# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0
#
#
# import numpy as np
# x=np.array([-1.0,1.0,2.0])
# print(x)
# y=x>0
# print(y)
# y=y.astype(np.int)
# print(y)


import numpy as np
import matplotlib.pylab as plt
#
# def step_function(x):
#     return np.array(x>0,dtype=np.int)
#
# x=np.arange(-5.0,5.0,0.1)
# y=step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))

# x=np.array([-1.0,1.0,2.0])
# print(sigmoid(x))
#
# x=np.arange(-5.0,5.0,0.1)
# y=sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()


# def relu(x):
#     return np.maximum(0,x)
#
# x=np.arange(-5.0,5.0,0.1)
# y=relu(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()


# A=np.array([1,2,3,4])
# print(A)
# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])


# B=np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)


# def identity_function(x):
#     return x
#
# def init_network():
#     network={}
#     network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
#     network['b1']=np.array([0.1,0.2,0.3])
#     network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
#     network['b2']=np.array([0.1,0.2])
#     network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
#     network['b3']=np.array([0.1,0.2])
#     return network
#
# def forward(network,x):
#     W1,W2,W3=network['W1'],network['W2'],network['W3']
#     b1,b2,b3=network['b1'],network['b2'],network['b3']
#
#     a1=np.dot(x,W1)+b1
#     z1=sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y=identity_function(a3)
#
#     return y
#
# network=init_network()
# x=np.array([1.0,0.5])
# y=forward(network,x)
# print(y)


# a=np.array([0.3,2.9,4.0])
# exp_a=np.exp(a)
# print(exp_a)
#
# sum_exp_a=np.sum(exp_a)
# print(sum_exp_a)
#
# y=exp_a/sum_exp_a
# print(y)
#
#
# def softmax(a):
#     exp_a=np.exp(a)
#     sum_exp_a=np.sum(exp_a)
#     y=exp_a/sum_exp_a
#
#     return y


def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

a=np.array([0.3,2.9,4.0])
y=softmax(a)
print(y)
print(np.sum(y))