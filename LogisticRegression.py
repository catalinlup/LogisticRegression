"""
This is an implementation of the Logistic Regression Algorithm.
It was created from scratch (no ML libraries used) by catalinlup.
Enjoy!
"""
import matplotlib.pyplot as plt
import csv
import math
from drawnow import*
import sys
import numpy as np


#Below are some functions and classes related to data storage, preprocessing and visualization

"""
The Dataset class contains two python lists: TrainingData and ExpectedOutput

TrainingData stores the data which will be used for training the model. In this case we store
the age and the estimated salary of each person as follows:
[[1,age1,salary1],
[1,age2,salary2],
...,
[1,age_n,salary_n]]
Note that the '1's' in the first column are used for simplicity reasons.
Instead of defining f(x)=a+b*x1+c*x2, where x1=age and x2=salary,
I defined f(x)=a*x0+b*x1+c*x2, where x0=1, x1=age and x2=salary

ExpectedOutput is a list of 1's and 0's based on whether or not the person purchased the object

To better understand:
        TrainingData:     |   ExpectedOutput:
Person1:[[1,age1,salary1],|    [1,
Person2: [1,age2,salary2],|     0,
...      ...              |    ...
PersonN: [1,ageN,salaryN]]|    1]


The ImportData function imports the data from the file

The NormalizeTrainingData function applies the Mean Normalization (element-mean)/(max-min)
on the dataset to make GradientDescent run smoother
"""
#The Dataset class
class Dataset:
    TrainingData=list()
    ExpectedOutput=list()
    def ImportData(self,filepath):
        file = open(filepath,'r')
        data=csv.reader(file,delimiter=',')
        for row in data:
            self.TrainingData.append([1,float(row[0]),float(row[1])])
            self.ExpectedOutput.append(float(row[2]))
    def NormalizeTrainingData(self):
        Mean=[0.0,0.0,0.0]
        Max=[0.0,0.0,0.0]
        Min=[math.inf,math.inf,math.inf]
        for row in self.TrainingData:
            Mean[1]+=row[1]
            Mean[2]+=row[2]
            Max[1]=max(Max[1],row[1])
            Max[2]=max(Max[2],row[2])
            Min[1]=min(Min[1],row[1])
            Min[2]=min(Min[2],row[2])
        ln=len(self.TrainingData)
        Mean[1]=Mean[1]/ln
        Mean[2]=Mean[2]/ln
        for i in range(0,ln):
            if Max[1]!=Min[1]:
                self.TrainingData[i][1]=(self.TrainingData[i][1]-Mean[1])/(Max[1]-Min[1])
            if Max[2]!=Min[2]:
                self.TrainingData[i][2]=(self.TrainingData[i][2]-Mean[2])/(Max[2]-Min[2])
    def getTrainingData(self):
        return self.TrainingData
    def getExpectedOutput(self):
        return self.ExpectedOutput
#End of the Dataset class
"""
The PlotTrainingData functions plots the training data on the screen using matplotlib
"""
def PlotTrainingData(TrainingData,ExpectedOutput):
    plt.figure(1)
    x0=list()
    y0=list()
    x1=list()
    y1=list()
    index=0
    for row in TrainingData:
        if ExpectedOutput[index]==0:
            x0.append(row[1])
            y0.append(row[2])
        else:
            x1.append(row[1])
            y1.append(row[2])
        index+=1
    plt.scatter(x0,y0,color='red',label='Not Purchased')
    plt.scatter(x1,y1,color='blue',label='Purchased')
    plt.legend(loc=2)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.show(block=True)

#End of the segment containing functions (and classes) related to data preprocessing
#and visualization





#Machine Learning related things
"""
Below is the definition of the logistic function {f(x)=1/(1+e^-x)}
it is also called 'sigmoid function'
"""
def LogisticFunction(z):#number as argument
    return 1/(1+math.exp(-z))
"""
The LinearFunction function basically represents the 'theta transposed * x'(from Andrew Ng's course)
which is equal to:
theta[0]*x[0]+theta[1]*x[1]+...+theta[N]*x[N]
"""
def LinearFunction(theta,x):#lists as argument
    if len(theta)!=len(x):
        print('Error Theta and x not same length')
        sys.exit()
    ln=len(theta)
    sum=0
    for i in range(0,ln):
        sum+=x[i]*theta[i]
    return sum
"""
This is the hypothesis of the ML algorithm
ht(x)=1/(1+e^-zt(x)) where zt(x)=theta[0]*x[0]+theta[1]*x[1]+...+theta[N]*x[N]
This hypothesis is often used in Logistic Regression
"""
def Hypothesis(theta,x):#vectors as argument
    return LogisticFunction(LinearFunction(theta,x))
"""
The Cost function is used as an auxilary in the definition of the CostFunction function
"""
def Cost(theta,x,y):
    if y==1:
        return -math.log(Hypothesis(theta,x))
    elif y==0:
        return -math.log(1-Hypothesis(theta,x))
"""
The CostFunction function computes (obviously) the cost function of the model
The CostFunction is at it's minimum when the model makes the best predictions based on the dataset
So our objective is to minimize it using GradientDescent
"""
def CostFunction(theta,TrainingData,ExpectedOutput):
    if len(TrainingData)!=len(ExpectedOutput):
        print('Error TrainingData and ExpectedOutput not same length')
        sys.exit()
    m=len(TrainingData)
    sum=0
    for i in range(0,m):
        sum=sum+(Cost(theta,TrainingData[i],ExpectedOutput[i]))/m
    return sum
"""
CostFunctionDerived computes the partial derivative of the CostFunction in respect to theta[j]
"""
def CostFunctionDerived(theta,TrainingData,ExpectedOutput,j):
    if len(TrainingData)!=len(ExpectedOutput):
        print('Error TrainingData and ExpectedOutput not same length')
        sys.exit()
    m=len(TrainingData)
    sum=0
    for i in range(0,m):
        sum=sum+((Hypothesis(theta,TrainingData[i])-ExpectedOutput[i])*TrainingData[i][j])/m
    return sum
"""
toDraw1,toDraw2,costPlot and makeFig are used for ploting the cost function graph
so don't bother with them
they are not essential for the algorithm
"""
toDraw1=[]
toDraw2=[]
costPlot=plt.figure(2)
def makeFig():
    plt.title('Cost Function Graph')
    plt.xlabel('iterations')
    plt.ylabel('Cost function value')
    plt.plot(toDraw2,toDraw1,label="Cost Function")
    plt.legend()
    costPlot.show()
"""
This function runs the GradientDescent algorithm
For more details check out Andrew Ng's course on ML
"""
def GradientDescent(TrainingData,ExpectedOutput,alfa,threshold):
    theta=[0,0,0]
    oldValue=0
    index=0
    while abs(oldValue-CostFunction(theta,TrainingData,ExpectedOutput))>threshold:
        oldValue=CostFunction(theta,TrainingData,ExpectedOutput)
        temp0=theta[0]-alfa*CostFunctionDerived(theta,TrainingData,ExpectedOutput,0)
        temp1=theta[1]-alfa*CostFunctionDerived(theta,TrainingData,ExpectedOutput,1)
        temp2=theta[2]-alfa*CostFunctionDerived(theta,TrainingData,ExpectedOutput,2)
        toDraw1.append(oldValue)
        toDraw2.append(index)
        drawnow(makeFig)
        theta[0]=temp0
        theta[1]=temp1
        theta[2]=temp2
        index+=1
    return theta
"""
This is used for plotting the decision Boundry
"""
def DecisionBoundry(theta,x):
    return -(theta[1]/theta[2])*x-(theta[0]/theta[2])
#end
"""
The code below calls some of the functions defined above in order to make the program run
You can see it as a 'main' function (for those who are familiar with C/C++,Java,etc)
Sorry for any misspelled words or grammar mistakes.
"""
if len(sys.argv)<5:
    print("Error! Please enter valid arguments!")
    sys.exit()
impDataPath=str(sys.argv[1])
outputDataPath=str(sys.argv[2])
alfa=float(sys.argv[3])
threshold=float(sys.argv[4])
data=Dataset()
data.ImportData(impDataPath)
data.NormalizeTrainingData()
theta=GradientDescent(data.getTrainingData(),data.getExpectedOutput(),alfa,threshold)#8, 0.0000003
file=open(outputDataPath,'w')
file.write(str(theta[0])+" "+str(theta[1])+" "+str(theta[2]))
t=np.arange(-0.5,0.5,0.1)
plt.figure(1)
plt.plot(t,DecisionBoundry(theta,t),'g-')
PlotTrainingData(data.getTrainingData(),data.getExpectedOutput())
