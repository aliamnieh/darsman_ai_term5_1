import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv('loans.csv')

# print(df.tail(50))

# print(df.dtypes)
# print(df.info)
# print(df.columns)
# print(df.describe())
# print(df.isnull().sum())


# this is a function for deleting outlier datas
# delete outlier data
def delete_outlier(df,column):
    min=df[column].min()
    max=df[column].max()
    q1=np.percentile(df[column],25)
    q3=np.percentile(df[column],75)
    iqr=q3-q1
    down_limit=q1-(1.5*iqr)
    up_limit=q3+(1.5*iqr)
    return df[(df[column]>down_limit)&(df[column]<up_limit)]

# converting the data type of these two columns
df.loan_start=pd.to_datetime(df.loan_start)
df.loan_end=pd.to_datetime(df.loan_end)

# making a new column that indicates the time length of the loan
df['loan_time_days']=df['loan_end']-df['loan_start']
df['loan_time_days']=pd.to_numeric(df['loan_time_days'])
# the time delta value is based on nano second, so we should convert it to day
df['loan_time_days']=df['loan_time_days'].apply(lambda x:(x/(24*3600*1000000000)))

# deleting the outlier datas
df=delete_outlier(df,'loan_amount')
df=delete_outlier(df,'rate')

# deleting the columns that we dont need in training the model
df.drop(['client_id','loan_id','loan_start','loan_end'],inplace=True,axis=1)

# one hot encoding for the categorical columns
df=pd.get_dummies(df,columns=['loan_type','repaid'])

scaler=StandardScaler()
df_scaled=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

# reseting the indexes of data frame
df_scaled.reset_index(inplace=True,drop=True)

# print(df_scaled)

df_scaled.to_csv('processed_data.csv')

