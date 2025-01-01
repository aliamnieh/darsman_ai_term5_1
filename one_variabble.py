import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# خواندن داده های پردازش شده
df_scaled=pd.read_csv('processed_data.csv')

# making the features matrix and target vector
y=df_scaled['loan_amount']
X=df_scaled.drop(['loan_amount'],axis=1)

# splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# ##########################################################################################################

# تبدیل فرمت داده ها به ماتریس
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#################################################################################################################

# ساخت مدل
model=LinearRegression()
model.fit(X_train[:,0].reshape(-1,1),y_train)

# مقایسه ی مقادیر اصلی و مقادیر پیش بینی شده
y_train_pred=model.predict(X_train[:,0].reshape(-1,1))
result1=np.hstack((y_train,y_train_pred))
mse=mean_squared_error(y_train,y_train_pred)
r2=r2_score(y_train,y_train_pred)
print(f'validating train data\n{20*'*'}')
print(f'mse = {mse:.3f}')
print(f'r2 = {r2:.3f}')
print(result1)

print(f'validating test data\n{20*'*'}')
y_test_pred=model.predict(X_test[:,0].reshape(-1,1))
result2=np.hstack((y_test,y_test_pred))
mse=mean_squared_error(y_test,y_test_pred)
r2=r2_score(y_test,y_test_pred)
print(f'mse = {mse:.3f}')
print(f'r2 = {r2:.3f}')
print(result2)