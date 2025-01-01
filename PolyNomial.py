import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# خواندن داده های پردازش شده
df=pd.read_csv('processed_data.csv')
# واکشی ستون های مورد نظر برای ساخت مدل چند چمله ای
y=df[['loan_amount']]
X=df[['rate','loan_time_days','loan_type_cash','loan_type_credit','loan_type_home','loan_type_other','repaid_0','repaid_1']]
# تبدیل فرمت داده ها به ماتریس
y=np.array(y)
X=np.array(X)

# تقسیم داده ها
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# ساخت مدل چند جمله ای
pf=PolynomialFeatures(degree=3)
X_train_poly=pf.fit_transform(X_train)
X_test_poly=pf.fit_transform(X_test)

model=LinearRegression()
model.fit(X_train_poly,y_train)

# مقایسه ی مقادیر اصلی با مقادیر پیش بینی شده
y_train_pred=model.predict(X_train_poly)
result=np.hstack((y_train,y_train_pred))
mse=mean_squared_error(y_train,y_train_pred)
r2=r2_score(y_train,y_train_pred)

print(f'validating train data\n{20*'*'}')
print(f'mse : {mse}')
print(f'r2 : {r2}')
print(result)

y_test_pred=model.predict(X_test_poly)
result=np.hstack((y_test,y_test_pred))
mse=mean_squared_error(y_test,y_test_pred)
r2=r2_score(y_test,y_test_pred)

print(f'validating test data\n{20*'*'}')
print(f'mse : {mse}')
print(f'r2 : {r2}')
print(result)