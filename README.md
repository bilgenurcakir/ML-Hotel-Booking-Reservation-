# ML Hotel Booking Reservation 
bu proje otel rezervasyon bilgilerini içeren bir veri setini kullanarak rezervasyon iptal etme olasılığını tahmin eden bir sınıflandırma projesidir.

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

data= pd.read_csv("hotel_bookings_updated_2024.csv")


#bu feature'lar tahminden sonra ortaya çıkan bilgileri içerdiğinden modelin doğru tahmin yapmasını engeller
silinecekler=["reservation_status","reservation_status_date"]

data=data.drop(columns=silinecekler)


#string ayları numeric hale getiririz. (ocak->1 subat->2 ...)
data["arrival_date_month"]=data["arrival_date_month"].apply(lambda x: datetime.strptime(x,'%B').month)
