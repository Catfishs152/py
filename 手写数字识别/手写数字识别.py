import pickle
import sys,os
sys.path.append(os.pardir)
from dataset.minist import load_mnist
import numpy as np
from PIL import Image

# def img_show(img):
#     pil_img=Image.fromarray(np.uint8(img))
#     pil_img.show()
#
# (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
#
# # print(x_train.shape)
# # print(t_train.shape)
# # print(x_test.shape)
# # print(t_test.shape)
#
# img=x_train[0]
# label=t_train[0]
# print(label)
#
# print(img.shape)
# img=img.reshape(28,28)
# print(img.shape)
#
# img_show(img)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y

x,t=get_data()
network=init_network()

batch_size=100
accuracy_cnt=0

for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

X,_=get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)

