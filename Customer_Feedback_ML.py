import os

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Veriyi okuma
customer_data = pd.read_csv("customer_feedback_satisfaction.csv")

# Cinsiyet kolonu: 2. kolonu LabelEncoder ile dönüştürme
le = preprocessing.LabelEncoder()
gender = customer_data.iloc[:, 2].values  # Cinsiyet kolonu (sadece tek boyutlu array)

# Cinsiyet sütununu sayısal hale getir (Male/Female -> 0/1)
gender_encoded = le.fit_transform(gender)  # Sayısal hale getirme

# OneHotEncoder uygulama (dönüştürülen sayısal cinsiyeti)
ohe = preprocessing.OneHotEncoder()
gender_ohe = ohe.fit_transform(gender_encoded.reshape(-1, 1)).toarray()  # OneHot kodlamayı 2D diziye çevir

# Gender'ı DataFrame'e dönüştürme
gender_df = pd.DataFrame(gender_ohe, columns=ohe.get_feature_names_out(['Gender']))

# Ülke (Country) kolonu: LabelEncoder ve OneHotEncoder
country = customer_data.iloc[:, 3].values
country_encoded = le.fit_transform(country)  # Sayısal hale getir
country_ohe = ohe.fit_transform(country_encoded.reshape(-1, 1)).toarray()  # OneHot kodlama

# Country'yi DataFrame'e dönüştürme
unique_country = ohe.get_feature_names_out(['Country'])
country_df = pd.DataFrame(country_ohe, columns=unique_country)

# FeedbackScore kolonu: LabelEncoder ve OneHotEncoder
feedbackscore = customer_data.iloc[:, 8].values
feedbackscore_encoded = le.fit_transform(feedbackscore)  # Sayısal hale getir
feedbackscore_ohe = ohe.fit_transform(feedbackscore_encoded.reshape(-1, 1)).toarray()  # OneHot kodlama

# FeedbackScore'u DataFrame'e dönüştürme
unique_feedbackscore = ohe.get_feature_names_out(['FeedbackScore'])
feedbackscore_df = pd.DataFrame(feedbackscore_ohe, columns=unique_feedbackscore)

# LoyaltyLevel kolonu: LabelEncoder ve OneHotEncoder
loyaltyLevel = customer_data.iloc[:, 9].values
loyaltyLevel_encoded = le.fit_transform(loyaltyLevel)  # Sayısal hale getir
loyaltyLevel_ohe = ohe.fit_transform(loyaltyLevel_encoded.reshape(-1, 1)).toarray()  # OneHot kodlama

# LoyaltyLevel'i DataFrame'e dönüştürme
unique_loyaltyLevel = ohe.get_feature_names_out(['LoyaltyLevel'])
loyaltyLevel_df = pd.DataFrame(loyaltyLevel_ohe, columns=unique_loyaltyLevel)

# İlk iki kolonu al
data1 = customer_data.iloc[:, :2].values
# Diğer kolonları al
data2 = customer_data.iloc[:, 4:8].values
data3 = customer_data.iloc[:, -1].values

# numpy array'lerini pandas DataFrame'e dönüştür
data1_df = pd.DataFrame(data1, columns=customer_data.columns[:2])
data2_df = pd.DataFrame(data2, columns=customer_data.columns[4:8])
data3_df = pd.DataFrame(data3, columns=[customer_data.columns[-1]])

# DataFrame'leri birleştirme
new_data1 = pd.concat([data1_df, gender_df, country_df], axis=1)
new_data2 = pd.concat([new_data1, data2_df, feedbackscore_df, loyaltyLevel_df, data3_df], axis=1)

# Sonuçları yazdırma
print(new_data2)
