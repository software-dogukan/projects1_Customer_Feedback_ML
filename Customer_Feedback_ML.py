import pandas as pd
import numpy as np
from sklearn import preprocessing

# Veriyi okuma
customer_data = pd.read_csv("customer_feedback_satisfaction.csv")

# Cinsiyet kolonu: 2. kolonu LabelEncoder ile dönüştürme
le = preprocessing.LabelEncoder()
gender = customer_data.iloc[:, 2:3].values  # Cinsiyet kolonu (DataFrame)
# Cinsiyeti sayısal hale getir
gender[:, 0] = le.fit_transform(gender[:, 0])
# OneHotEncoder uygulama (dönüştürülen sayısal cinsiyeti)
ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()  # OneHot kodlamayı 2D diziye çevir


country=customer_data.iloc[:,3].values
country=le.fit_transform(country)
country=ohe.fit_transform(country.reshape(-1, 1)).toarray()
unique_words = customer_data['Country'].unique()
country=pd.DataFrame(country,columns=[unique_words])
print(country)

#for feedbackscore
feedbackscore=customer_data.iloc[:,8].values
feedbackscore=le.fit_transform(customer_data["FeedbackScore"])
feedbackscore=ohe.fit_transform(feedbackscore.reshape(-1, 1)).toarray()
unique_words = le.classes_
# One-hot encoding sonucu elde edilen feedbackscore verilerini pandas DataFrame'e dönüştürme
feedbackscore_df = pd.DataFrame(feedbackscore, columns=unique_words)


loyaltyLevel=customer_data.iloc[:,9].values
loyaltyLevel=le.fit_transform(customer_data["LoyaltyLevel"])
loyaltyLevel=ohe.fit_transform(loyaltyLevel.reshape(-1, 1)).toarray()
unique_words = le.classes_
# One-hot encoding sonucu elde edilen feedbackscore verilerini pandas DataFrame'e dönüştürme
loyaltyLevel_df = pd.DataFrame(loyaltyLevel, columns=unique_words)
print(loyaltyLevel_df)



# İlk iki kolonu al
data1 = customer_data.iloc[:, :2].values
# Diğer kolonları al
data2 = customer_data.iloc[:, 4:8].values
data3=customer_data.iloc[:,-1].values
# numpy array'lerini pandas DataFrame'e dönüştür

data1_df = pd.DataFrame(data1, columns=customer_data.columns[:2])
gender_df = pd.DataFrame(gender, columns=ohe.get_feature_names_out([customer_data.columns[2]]))
data2_df = pd.DataFrame(data2, columns=customer_data.columns[3:8])
data3_df = pd.DataFrame(data3, columns=customer_data.columns[-1])
# DataFrame'leri birleştirme
new_data1 = pd.concat([data1_df, gender_df.iloc[:,0],country], axis=1)
new_data2=pd.concat([new_data1,data2_df,feedbackscore_df,loyaltyLevel_df,data3_df],axis=1)
print(new_data2)







