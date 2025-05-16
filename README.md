# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values
4. Find the accuracy and display the result.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NITHYASRI M
RegisterNumber:  212224040226
*/
import chardet
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv( "/content/spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

## Output:
![image](https://github.com/user-attachments/assets/93aa0d88-bc23-494e-9a77-73094cf9c771)
![image](https://github.com/user-attachments/assets/0f9e3c50-a6bd-4b8a-b3b6-50a1ebdd94cb)
![image](https://github.com/user-attachments/assets/f625a903-d999-4898-8a54-d26bcffc807d)
![image](https://github.com/user-attachments/assets/ad955f9c-bff4-4c62-9db2-5cfa429b8f1a)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
