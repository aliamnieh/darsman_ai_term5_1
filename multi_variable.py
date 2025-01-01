import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# خواندن داده های پردازش شده
df_scaled=pd.read_csv('processed_data.csv')

# تبدیل فرمت داده ها به ماتریس
X=df_scaled[['rate','loan_time_days']]
y=df_scaled[['loan_amount']]
X=np.array(X)
y=np.array(y)

# تقسیم داده ها
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# ساخت مدل
model=LinearRegression()
# اموزش مدل
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)

# مقایسه ی داده های اصلی و داده های پیش بینی شده
result=np.hstack((y_train,y_train_pred))
# محاسبه ی میزان خطا
mse=mean_squared_error(y_train,y_train_pred)
r2=r2_score(y_train,y_train_pred)

print(f'validating train data\n{20*'*'}')
print(f'mse : {mse}')
print(f'r2 : {r2}')
print(result)


y_test_pred=model.predict(X_test)
# مقایسه ی مقادیر اصلی و مقادیر پیش بینی شده برای داده های تست
result=np.hstack((y_test,y_test_pred))
mse=mean_squared_error(y_test,y_test_pred)
r2=r2_score(y_test,y_test_pred)
print(f'validating test data\n{20*'*'}')
print(f'mse : {mse}')
print(f'r2 : {r2}')
print(result)