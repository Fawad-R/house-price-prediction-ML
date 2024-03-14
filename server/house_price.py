from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
readCsv=r'F:\Artificial Intelligence\Machine Learning\3_ML Playlist\Machine Learning Projects\1_Deployement\House price prediction\server\housing.csv'
df=pd.read_csv(readCsv)
year=df['ocean_proximity'].value_counts()
a=year[year<=6]
df['ocean_proximity']=df['ocean_proximity'].apply(lambda x: 'INLAND' if x in a else x)
newdf=df
OHE=OneHotEncoder()
newdf['ocean_proximity']=OHE.fit_transform(newdf[['ocean_proximity']]).shape[0]
newdf['total_bedrooms'].fillna(newdf['total_bedrooms'].mean(),inplace=True)
newdf['total_rooms']=newdf['total_rooms'].astype(int)
newdf['median_house_value']=newdf['median_house_value'].astype(int)
newdf['households']=newdf['households'].astype(int)
newdf['housing_median_age']=newdf['housing_median_age'].astype(int)
Q1=newdf['total_rooms'].quantile(0.25)
Q3=newdf['total_rooms'].quantile(0.75)
IQR=Q3-Q1
min=Q1-1.5*IQR
max=Q3+1.5*IQR
newdf['total_rooms']=np.where(newdf['total_rooms']>max,max,np.where(newdf['total_rooms']<min,min,newdf['total_rooms']))
Q1=newdf['total_bedrooms'].quantile(0.25)
Q3=newdf['total_bedrooms'].quantile(0.75)
IQR=Q3-Q1
min=Q1-1.5*IQR
max=Q3+1.5*IQR
newdf['total_bedrooms']=np.where(newdf['total_bedrooms']>max,max,np.where(newdf['total_bedrooms']<min,min,newdf['total_bedrooms']))
Q1=newdf['population'].quantile(0.25)
Q3=newdf['population'].quantile(0.75)
IQR=Q3-Q1
min=Q1-1.5*IQR
max=Q3+1.5*IQR
newdf['population']=np.where(newdf['population']>max,max,np.where(newdf['population']<min,min,newdf['population']))
Q1=newdf['households'].quantile(0.25)
Q3=newdf['households'].quantile(0.75)
IQR=Q3-Q1
min=Q1-1.5*IQR
max=Q3+1.5*IQR
newdf['households']=np.where(newdf['households']>max,max,np.where(newdf['households']<min,min,newdf['households']))
Q1=newdf['median_income'].quantile(0.25)
Q3=newdf['median_income'].quantile(0.75)
IQR=Q3-Q1
min=Q1-1.5*IQR
max=Q3+1.5*IQR
newdf['median_income']=np.where(newdf['median_income']>max,max,np.where(newdf['median_income']<min,min,newdf['median_income']))
x=newdf.drop(columns=['median_house_value'])
y=newdf['median_house_value']
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
OHE=OneHotEncoder(sparse=False)
scaler=MaxAbsScaler()
xtrain=scaler.fit_transform(xtrain)
xgb = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb.fit(xtrain,ytrain)
def Predict_Price(xtest):
    xtest=scaler.transform(xtest)
    ypred=xgb.predict(xtest)
    return ypred

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    response = Predict_Price([[data['longitude'],data['latitude'],data['housing_median_age'],data['total_rooms'],data['total_bedrooms'],data['population'],data['households'],data['median_income'],data['ocean_proximity']]])
    response = np.array(response)
    response_list = response.tolist()
    return jsonify({'response': response_list})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)        