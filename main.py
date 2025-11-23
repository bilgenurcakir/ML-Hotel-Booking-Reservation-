import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from datetime import datetime


data=pd.read_csv("hotel_bookings_updated_2024.csv")

silinecekler=["reservation_status","reservation_status_date"] #bu featureslar tahmin sonrası
# için değerler verdiğinden dolayı tahmini hatalı sonuçlandıracaktır.
data=data.drop(columns=silinecekler)

print(data.head())

data['arrival_date_month']=data['arrival_date_month'].apply(lambda x: datetime.strptime(x,'%B').month) #  string ay değerini sayı ile yeniden yaz

onehot_cols=[
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "city",
    "country",
]# one hot encoder ile numeric yapmamız gereken sütunlar

data=pd.get_dummies(data,columns=onehot_cols,drop_first=True) #onehot encoding




print("onehot sonrası kolon sayısı:",len(data))
print(data.head())

numerical_columns=data.select_dtypes(include=np.number).columns #numeric sütunları al

for c in numerical_columns:
    if c!='is_canceled':
         print( f" {c} ile target arası korelasyon:{data['is_canceled'].corr(data[c])}") # tüm numeric sütunlar için is_canceled (target) ile korelasyonuna bak.


gereksiz_sutunlar=[
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
]# korelasyon sonrasında gereksiz görülen sütunlar


data=data.drop(columns=gereksiz_sutunlar)
data = data.fillna(0) # geriye kalan sütunlarda nan değerlerini 0 ile dolduralım




x=data.drop('is_canceled',axis=1)
y=data['is_canceled']

x_train ,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=True)

print(data.info())

#1-) logistic regression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
tahmin=logmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print(" logistic regression doğruluk skoru: ",accuracy)



dtmodel=DecisionTreeClassifier()
dtmodel.fit(x_train,y_train)
tahmin=dtmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)

print(" decisition trees doğruluk skoru: ",accuracy)

rfmodel=RandomForestClassifier()
rfmodel.fit(x_train,y_train)
tahminrf=rfmodel.predict(x_test)

accuracyrf=accuracy_score(y_test,tahminrf)
print("random forest doğruluk skoru:",accuracyrf)


knnmodel=KNeighborsClassifier()
knnmodel.fit(x_train,y_train)
tahmin=knnmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print("KNN doğruluk skoru:",accuracy)

'''
 bu boyutta bir veri için svc kullanılmaz.
aşırı yavaş olduğu için kullanmadım.
svcmodel=SVC(random_state=42)
svcmodel.fit(x_train,y_train)
tahmin=svcmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print("svc doğruluk skoru:",accuracy)
'''
print("svc doğruluk skoru: skor heaplanamadı çünkü bu yükseklikte bir veri için svr çok yavaş çalışmaktadır.")


feature_importance = pd.Series(
    rfmodel.feature_importances_,
    index=x_train.columns
).nlargest(10).sort_values(ascending=True)

# Görselleştirme
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='skyblue')

plt.title('Random Forest - En Önemli 10 Özellik')
plt.xlabel('Önem Derecesi')
plt.grid(axis='x', linestyle='--')
plt.show()


print("\n en yüksek doğruluğu veren model: Random Forest modelidir.")
print("Accuracy (train)  %0.1f " % (accuracyrf * 100))
print(classification_report(y_test, tahminrf))
