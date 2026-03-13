# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M kiruthika vasanthi
RegisterNumber:  212225040189
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img width="894" height="603" alt="562892906-b1467da4-b969-44d4-96b0-83397974370f" src="https://github.com/user-attachments/assets/b5341f39-fdad-4410-afde-df62379e1d5b" />

<img width="873" height="146" alt="562892953-5f9d10aa-8f5b-4365-9589-810600f38b39" src="https://github.com/user-attachments/assets/ddc0ccf0-3472-41ee-994b-42e6941f5b95" />

<img width="878" height="157" alt="562893018-aee3ab33-412e-447d-ab55-fef149f1b788" src="https://github.com/user-attachments/assets/3de5cd71-52ea-457d-900f-c95eb3e3a6ed" />

<img width="895" height="341" alt="562893074-2fce8f49-2399-43bc-9009-eb826138ddf9" src="https://github.com/user-attachments/assets/04d483f3-aeef-4f15-943e-0a394d8253d4" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
