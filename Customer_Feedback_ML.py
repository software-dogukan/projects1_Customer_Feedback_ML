import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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

# Bağımsız değişkenler (x) ve bağımlı değişken (y)
left_x = new_data2.iloc[:, 1:9].values
right_x = new_data2.iloc[:, 10:].values

left_x_df = pd.DataFrame(left_x, columns=new_data2.columns[1:9])
right_x_df = pd.DataFrame(right_x, columns=new_data2.columns[10:])

# x ve y'yi birleştir
x = pd.concat([left_x_df, right_x_df], axis=1)
y = new_data2.iloc[:, 9].values

# Eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Sklearn ile doğrusal regresyon
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Test setinde tahmin yapma
y_pred = regressor.predict(x_test)

# Statsmodels ile doğrusal regresyon modeli oluşturma
# Sabit sütun eklemek için
ones = np.ones((x.shape[0], 1))  # x'in satır sayısı kadar 1'lerle bir sütun ekliyoruz
X = np.append(ones, x, axis=1)   # Sabit sütun ile x'i birleştiriyoruz

# Statsmodels modelini kurma
model = sm.OLS(y, X).fit()  # OLS modelini fit etme

# Model özetini yazdırma
print(model.summary())
