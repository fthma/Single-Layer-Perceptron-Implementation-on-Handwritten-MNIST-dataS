#!/usr/bin/env python
# coding: utf-8

# In[335]:


from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score

#filepath=os.path.dirname(os.path.realpath(_file_))
directorypath=os.getcwd()
print(directorypath)
filepath=os.path.join(directorypath,'train-images.idx3-ubyte')
print(filepath)
# In[336]:
"""

TrainData, TrainLabel = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')
TestData, TestLabel = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')


# In[337]:


def getData(Data,Labels,num):
    X=Data[:num]
    D=Labels[:num]
    return np.asarray(X),np.asarray(D)

#getting 500 trainig data and 100 testing data
X,Labels=getData(TrainData,TrainLabel,500)
Test_X_100,Test_Labels_100=getData(TestData,TestLabel,100)  

#getting 5000 trainig data and 500 testing data
X_Train_5000,D_Train_5000=getData(TrainData,TrainLabel,5000)
X_Test_500,D_Test_500=getData(TestData, TestLabel,500)


# In[338]:


def createLabelArray(Labels):
    D=[]
    for i in Labels:
        d=np.zeros(10,order='F')
        d[i]=1
        D.append(d)
    D=np.array(D)
    D=np.transpose(D)
    return D

D=createLabelArray(Labels)
TestD=createLabelArray(Test_Labels_100)

D_5000=createLabelArray(D_Train_5000)
TestD_500=createLabelArray(D_Test_500)


# In[339]:


def normalization(XArr):
    x_train=[]
    for x in XArr:
        retval, threshold = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
        threshold=threshold.ravel()
        threshold=np.asarray(threshold).reshape(784,1)
        x_train.append(threshold/255)
        
    return np.asarray(x_train)

#Normalizing 500 training data and 100 testing data
x_train=normalization(X)
x_test=normalization(Test_X_100)

#Normalizing 5000 training data and 500 testing data
x_train_5000=normalization(X_Train_5000)
x_test_500=normalization(X_Test_500)


# In[349]:


def SLP_Training(Xi,Di,Beta=0.00025,sigmoid=False):
    #initialize weights
    print("Traning begins")
    W=np.random.random_sample((10,784))
    trainingData=Xi
    initW=W
    if(sigmoid==True):
        Ebselon=0.015
    else:
        Ebselon=0.01
    #Ebselon=0.0000000001
    num=len(Xi)
    X_Trans=np.transpose(trainingData).reshape(784,num)
    MSE=[]
    mse=1
    print("Inside Training while loop")
    while(mse>Ebselon):
        V=np.matmul(W,X_Trans)
        if(sigmoid==False):
            Y=np.where(V<0, 0, 1)
        else:
            Y=1/(1+np.exp(-V))
        E=Di-Y
        #insert code to check E[j] with Ebselon
        Delta_W=Beta*np.matmul(E,Xi.reshape(num,784))
        New_W=W+Delta_W
        W=New_W
        mse=np.power(E,2).mean()
        MSE.append(mse)    
    
    return MSE,W


# In[353]:


MSE,Weights=SLP_Training(x_train,D)
plt.plot(np.arange(len(MSE)),MSE)
plt.xlabel('iterations')
plt.ylabel('Mean Square Error ')
plt.title('Mean Square Error Vs iterations')
plt.show()


# In[342]:


def testing(X,D,W,Beta):
    X_Trans=np.transpose(X)
    num=len(X)
    X_Trans=X_Trans.reshape(784,num)
       
    print("Testing begins")
    V=np.matmul(W,X_Trans)
    Y= np.where(V >=0 ,1 , 0)
    E=D-Y
    ErrorCount=[]
    ExactDigitOccurence=[]
    
    print("Inside Testing  loop")
    for i in range(len(E)):
        ErrorCount.append(np.count_nonzero(E[i]==1))
        ExactDigitOccurence.append(np.count_nonzero(D[i]==1))
                
    PercentageError=np.asarray(ErrorCount)/np.asarray(ExactDigitOccurence)*100
    mse=np.power(E,2).mean()

    return PercentageError


# In[343]:


def draw_barCharts(x_train,D,x_test,TestD,sig=False):
    Beta1=0.5
    Beta2=0.05
    Beta3=0.005
    
    #learning rate beta=0.5
    MSE,Weights=SLP_Training(x_train,D,Beta=Beta1,sigmoid=sig)
    PercentageError1=testing(x_test,TestD,Weights,Beta1)
    
    Label1="learningRate={}".format(Beta1)
    plt.bar(np.arange(10),PercentageError1,label=Label1, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta1))
    plt.show()
    
    #learning rate beta=0.05
    MSE,Weights=SLP_Training(x_train,D,Beta=Beta2,sigmoid=sig)
    PercentageError2=testing(x_test,TestD,Weights,Beta2)
    
    Label2="learningRate={}".format(Beta2)
    plt.bar(np.arange(10),PercentageError2,label=Label2, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta2))
    plt.show()
    
    #learning rate beta=0.005
    MSE,Weights=SLP_Training(x_train,D,Beta=Beta3,sigmoid=sig)
    PercentageError3=testing(x_test,TestD,Weights,Beta3)
      
    Label3="learningRate={}".format(Beta3)
    plt.bar(np.arange(10),PercentageError3,label=Label3, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta3))
    plt.show()
    
    


# In[344]:


#task 1
draw_barCharts(x_train,D,x_test,TestD)


# In[345]:


#task 2
#Since it is a BIG Dataset takes about 20 secs to train and display for 3 different learning rates
print("Task 2: Training-5000, Testing-500")

draw_barCharts(x_train_5000,D_5000,x_test_500,TestD_500)


# In[350]:


#Task 3
#normalizing the 5000 trainig dataset withould thresholding
task3_X=X_Train_5000/255
draw_barCharts(task3_X,D_5000,x_test_500,TestD_500,sig=True)

"""
