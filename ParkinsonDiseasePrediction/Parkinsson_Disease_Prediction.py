#define
#This program indicates if individuals has parkinson's disease or not
#parkinson disease is a progressive nervous system disorder that affects movement
import  pandas as pd
import  numpy as np
from    xgboost import XGBClassifier
from    sklearn.preprocessing import MinMaxScaler
from    sklearn.metrics import classification_report
from    sklearn.model_selection import train_test_split


df=pd.read_csv("Parkinsson disease.csv")
#print(df.head())

#print(df.isnull().values.any())


#print(df.shape) 
print(df['status'].value_counts())
percentage_has_disease=147/(48+147)*100
percentage_donthas_disease=48/(147+48)*100
print("Have Disease Percentage:",percentage_has_disease,"%","Dont Have Disease Percentage",percentage_donthas_disease,"%")

#data types
print(df.dtypes)

#feature data set

x=df.drop(['name'],1)
x=np.array(x.drop(['status'],1))

#target dataset
y=np.array(df['status'])

# 80% training and 20% testing data sets
x_train,x_test ,y_train, y_test = train_test_split(x,y,test_size=0.3)

#transform data to 0 ,1
sc=MinMaxScaler(feature_range=(0,1))
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)

#XGBClassifier
XGBClassifier().use_label_encoder=False
model=XGBClassifier().fit(x_train,y_train)
model.use_label_encoder=False
predictions=model.predict(x_test)
print(predictions)
print(y_test)
print(classification_report(y_test,predictions))

