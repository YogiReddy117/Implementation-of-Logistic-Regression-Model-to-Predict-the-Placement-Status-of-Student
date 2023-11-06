# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Palleri Yogi
RegisterNumber:  212220040108
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Placement Data:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/26afa571-4852-4b97-9b71-c31b4e341fe5)

Salary Data:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/aeb57499-f9b7-4359-9a17-3d4e1b30f447)

Checking the null() function:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/3e3b092a-ac1e-4f76-bc69-127b4a7912ed)

Data Duplicate:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/7923aa6a-8152-436e-84b8-d30b43a5885e)

Print Data:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/e802586e-ba7b-43e1-97a3-1e45d902792f)

Data-Status:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/3cd4767a-f4d5-4074-b529-0140ec107473)

Y_prediction array:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/ec198641-df94-4156-b9b1-ebbe993cb880)

Accuracy value:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/71ab14ca-fc99-496b-8ec0-d4b45f7f89f3)

Confusion array:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/5c7fb12b-1d98-48cd-8891-5f46d539d85a)

Classification Report:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/a81463b3-fcd1-4b40-8a49-00baa7098835)

Prediction of LR:

![image](https://github.com/YogiReddy117/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123739437/42b3e3f2-42e4-4eca-98ea-089bd2bba0ca)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
