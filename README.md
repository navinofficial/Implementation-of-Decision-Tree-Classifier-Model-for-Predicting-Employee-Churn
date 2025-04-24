# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Navinkumar v
RegisterNumber: 212223230141
*/

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
## Dataset
![image](https://github.com/user-attachments/assets/eb48b7d5-c4f3-425a-9417-91a0e5a96f7a)
## Information
![image](https://github.com/user-attachments/assets/d6c4b766-05c0-489a-987e-5e2785330c5e)
## Non Null values
![image](https://github.com/user-attachments/assets/254f47d7-4bd9-453a-8a60-32c4f3a79eb1)
## Encoded value
![image](https://github.com/user-attachments/assets/2fe97da7-3325-4d5c-9abc-1ae0aac16b18)
## Count
![image](https://github.com/user-attachments/assets/c28ca954-74f8-4d8c-92ba-578b6adf62d7)
## X and Y value
![image](https://github.com/user-attachments/assets/e2dece95-a692-4126-97ee-e680601e88fb)
![image](https://github.com/user-attachments/assets/825390e2-3be9-4937-ac39-5efbec235e03)
## Accuracy
![image](https://github.com/user-attachments/assets/61d3ada6-1cb6-4e87-a4ac-106f18b0a1ec)
## Predicted
![image](https://github.com/user-attachments/assets/7f458ead-87d3-4422-8940-28ed6c1e3c77)


## Plot
![DT](https://github.com/user-attachments/assets/c2d49522-4306-4457-a125-44b75464882f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
