# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Keziah.F
RegisterNumber:  212223040094
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```
## Output:
## DATA HEAD 
![image](https://github.com/user-attachments/assets/c1a141b9-c814-4a1f-81dc-2f824de420bc)

## DATA INFO

![image](https://github.com/user-attachments/assets/51495eee-7de5-4713-ab0c-211204e60217)

## DATA HEAD FOR SALARY

![image](https://github.com/user-attachments/assets/0eb62ecb-a329-4fb4-b711-058c04c16dd5)

## MEAN SQUARED ERROR

![image](https://github.com/user-attachments/assets/80e77d50-f598-4a31-b958-854d23c56063)

## R2 VALUE

![image](https://github.com/user-attachments/assets/2a45f6e9-33d7-4751-bbbd-45fb7349affa)

## DATA PREDICTION

![image](https://github.com/user-attachments/assets/1bbcb0f0-4bb2-4bce-aa37-059320cf7ebd)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
