# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM :

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### Step 1 :

Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.

### Step 2 :

Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.

### Step 3 :

Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.

### Step 4 :

Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.

### Step 5 :

Predict the value of y and also print the values of m and c.

### Step 6 :

Plot the accquired graph with respect to hours and scores using the scatter plot function.

### Step 7 :

End the program.

## Program :

### Program to implement the linear regression using gradient descent.

### DEVELOPED BY : ABRIN NISHA A
### REG NO : 212222230005

```
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model

  """
  m=len(y)  
  h=X.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  """
   Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output :


### PROFIT PREDICTION :

![Screenshot 2023-09-14 102434](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/fe54ded8-d464-4d06-8187-3954a649827e)

### COST FUNCTION :

![Screenshot 2023-09-14 102536](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/6161855f-7f07-4145-b3e5-627b007a7c06)

### GRADIENT DESCENT :

![Screenshot 2023-09-14 102600](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/91982e85-620e-4587-afd4-7866a6481dce)

### COST FUNCTION USING GRADIENT DESCENT :

![Screenshot 2023-09-14 102609](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/a1601e7d-7bae-4935-bfa1-16afd4c985c7)

### GRAPH WITH BEST FIT LINE (PROFIT PREDICTION) :

![Screenshot 2023-09-14 102636](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/7a115b20-d4e3-4b4e-b645-2959b8e28dea)

### PROFIT PREDICTION FOR A POPULATION OF 35,000 & 70,000 :

![Screenshot 2023-09-14 102647](https://github.com/Abrinnisha6/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118889454/f1a8687d-7605-4056-b234-5728bb3da6c5)

## Result :

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
