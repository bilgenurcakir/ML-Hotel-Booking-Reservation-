# ML Hotel Booking Reservation 
bu proje otel rezervasyon bilgilerini içeren bir veri setini kullanarak rezervasyon iptal etme olasılığını tahmin eden bir sınıflandırma projesidir.

# nasıl çalışır?
öncelikle veri setimizin içindeki modelleri yanlış etkilecek olan features'lar çıkarıldı(örneğin burada reservation_status hali hazırda iptal edilip edilmediği bilgisini içeriyordu)

daha sonrasında featureslardan biri olan arrival_date_month satırını daha iyi kullanabilmek için numerical hale getirme ihtiyacı duyduk ve her ay için (ocak->1, şubat->2..) string ifadeyi sayısal karşılıklarını verdik.

veri setindeki özelliklerin is_canceled(target) ile korelasyonlarını bulabilmek için kategorik verileri numeric verilere çevirdik(onehot encoding kullandık çünkü ordinal bir feature yok).

numeric sütunları seçerek bu sütunlar üzerinde her biri için target ile korelasyon hesabı yaptık.
bu hesaplamada korelasyonu düşük olan ve modelin performansını düşürecek olan sütunları belirledik ve bu sütunları drop ile veri setinden attık.

data içerisindeki Nan değerleri 0 ile doldurduk.

girdi ve çıktımızı belirledik ve veri setimizi test ve train olarak böldük.

teker teker sınıflandırma için kullanabileceğimiz modelleri denedik( Logistic regression, decisionTree, RandomForest, SVC,KNN)
bu modellerden SVC bu büyüklükte bir veri seti için çok uzun sürdüğünden eledik.
geriye kalanlardan da en başarılısı RandomForest oldu(acc=87)

en önemli 10 feature ekrana tablolaştırıldı ve en son konsola accuricy report bastırıldı.







# kütüphaneler

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


# kodun açıklamaları

### veri setini çekeriz

data= pd.read_csv("hotel_bookings_updated_2024.csv") 

### bu feature'lar tahminden sonra ortaya çıkan bilgileri içerdiğinden modelin doğru tahmin yapmasını engeller

silinecekler=["reservation_status","reservation_status_date"]

data=data.drop(columns=silinecekler) #o satırları siler.

### veri setinin ilk 5 satırını bastırır.
print(data.head())

### string ayları numeric hale getiririz. (ocak->1 subat->2 ...)
data["arrival_date_month"]=data["arrival_date_month"].apply(lambda x: datetime.strptime(x,'%B').month)

### onehot encoder ile numerik yapılası gereken catagorical featuresları seçeriz.
onehot_cols=["hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "city",
    "country",
] 

### onehot encoder
data=pd.get_dummies(data,columns=onehot_cols,drop_first=True) 

print("onehot sonrası kolon sayısı:",len(data))
print(data.head())

### numeric sütunları seçeriz
numerical_columns=data.select_dtypes(include=np.number).columns 
for c in numerical_columns:
    if c!='is_canceled':
         print( f" {c} ile target arası korelasyon:{data['is_canceled'].corr(data[c])}") 
         # tüm numeric sütunlar için is_canceled (target) ile korelasyonuna bak.

### bu aşamada kodu çalıştırarak korelasyonlarını kontrol ettik, model için gereksiz/performansını düşürecek olan featuresları belirledik.

### korelasyon sonrasında gereksiz görülen sütunlar
gereksiz_sutunlar=[
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'stays_in_weekend_nights',
]


data=data.drop(columns=gereksiz_sutunlar) #gereksiz sutünları attık.

### geriye kalan sütunlarda nan değerlerini 0 ile dolduralım
data = data.fillna(0)

### girdi ve çıktı
x=data.drop('is_canceled',axis=1) # target haricindeki sutünlar
y=data['is_canceled'] # target

### test ve train olarak veri setini ayırdık
x_train ,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=True)

print(data.info())

### modeller

#### 1-) logistic regression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
tahmin=logmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print(" logistic regression doğruluk skoru: ",accuracy)


#### 2-) DecisionTree
dtmodel=DecisionTreeClassifier()
dtmodel.fit(x_train,y_train)
tahmin=dtmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)

print(" decisition trees doğruluk skoru: ",accuracy)

#### 3-)RandomForest
rfmodel=RandomForestClassifier()
rfmodel.fit(x_train,y_train)
tahminrf=rfmodel.predict(x_test)

accuracyrf=accuracy_score(y_test,tahminrf)
print("random forest doğruluk skoru:",accuracyrf)

#### 4-)KNN
knnmodel=KNeighborsClassifier()
knnmodel.fit(x_train,y_train)
tahmin=knnmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print("KNN doğruluk skoru:",accuracy)

'''
#### 5-)SVC
##### bu boyutta bir veri için svc kullanılmaz (veri seti çok büyük).
aşırı yavaş olduğu için kullanmadım.

svcmodel=SVC(random_state=42)
svcmodel.fit(x_train,y_train)
tahmin=svcmodel.predict(x_test)

accuracy=accuracy_score(y_test,tahmin)
print("svc doğruluk skoru:",accuracy)
'''
print("svc doğruluk skoru: skor heaplanamadı çünkü bu yükseklikte bir veri için svr çok yavaş çalışmaktadır.")

### en etkili 10 özellik ekrana tablolaştırılır.
feature_importance = pd.Series(
    rfmodel.feature_importances_,
    index=x_train.columns
).nlargest(10).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='skyblue')

plt.title('Random Forest - En Önemli 10 Özellik')
plt.xlabel('Önem Derecesi')
plt.grid(axis='x', linestyle='--')
plt.show()

### son aşama, burada kod çalıştırılarak en yüksek doğruluğa sahip olan model RandomForest olarak seçildi,
accuricy report bastırıldı.

print("\n en yüksek doğruluğu veren model: Random Forest modelidir.")

print("Accuracy (train)  %0.1f " % (accuracyrf * 100))

print(classification_report(y_test, tahminrf))

# model çıktısı

#ilk data.head()
                       hotel  ...        city
0  Resort Hotel - Chandigarh  ...  Chandigarh
1      Resort Hotel - Mumbai  ...      Mumbai
2       Resort Hotel - Delhi  ...       Delhi
3     Resort Hotel - Kolkata  ...     Kolkata
4     Resort Hotel - Lucknow  ...     Lucknow

[5 rows x 31 columns]
onehot sonrası kolon sayısı: 119390

#onehot sonrası data.head()
   is_canceled  lead_time  ...  country_ZMB  country_ZWE
0            0        342  ...        False        False
1            0        737  ...        False        False
2            0          7  ...        False        False
3            0         13  ...        False        False
4            0         14  ...        False        False

[5 rows x 280 columns]

#korelasyonlar:
 lead_time ile target arası korelasyon:0.29312335576070536
 
 arrival_date_year ile target arası korelasyon:nan
 
 arrival_date_month ile target arası korelasyon:0.0005832006134483165
 
 arrival_date_week_number ile target arası korelasyon:0.0005000172975428105
 
 arrival_date_day_of_month ile target arası korelasyon:-0.0038576620217618644
 
 stays_in_weekend_nights ile target arası korelasyon:-0.001791078078260647
 
 stays_in_week_nights ile target arası korelasyon:0.024764629045871265
 
 adults ile target arası korelasyon:0.060017212839559804
 
 children ile target arası korelasyon:0.0050477900292686196
 
 babies ile target arası korelasyon:-0.03249108920833171
 
 is_repeated_guest ile target arası korelasyon:-0.08479341835708092
 
 previous_cancellations ile target arası korelasyon:0.11013280822282377
 
 previous_bookings_not_canceled ile target arası korelasyon:-0.057357723165940115
 
 booking_changes ile target arası korelasyon:-0.1443809910613153
 
 agent ile target arası korelasyon:-0.08311415905369401
 
 company ile target arası korelasyon:-0.02064207062825651
 
 days_in_waiting_list ile target arası korelasyon:0.05418582411777437
 
 adr ile target arası korelasyon:0.047556597880384174
 
 required_car_parking_spaces ile target arası korelasyon:-0.1954978174944898
 
 total_of_special_requests ile target arası korelasyon:-0.2346577739690115

#data.info()
RangeIndex: 119390 entries, 0 to 119389

Columns: 275 entries, is_canceled to country_ZWE

dtypes: bool(259), float64(4), int64(12)

memory usage: 44.1 MB

None

#model sonuçları
 logistic regression doğruluk skoru:  0.8004858028310579
 
 decisition trees doğruluk skoru:  0.8426166345590083

random forest doğruluk skoru: 0.8730211910545271

KNN doğruluk skoru: 0.78478096993048

svc doğruluk skoru: skor heaplanamadı çünkü bu yükseklikte bir veri için svr çok yavaş çalışmaktadır.


 en yüksek doğruluğu veren model: Random Forest modelidir.
 
Accuracy (train)  87.3 
              precision    recall  f1-score   support

           0       0.88      0.93      0.90     15101
           1       0.87      0.77      0.82      8777

    accuracy                           0.87     23878
   macro avg       0.87      0.85      0.86     23878
weighted avg       0.87      0.87      0.87     23878


Process finished with exit code 0



