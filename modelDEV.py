import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "engine_size": [1.2, 1.5, 1.6, 2.0, 2.2, 3.0, 1.8, 2.5],
    "horsepower": [85, 100, 110, 150, 170, 250, 130, 200],
    "car_age": [8, 6, 5, 4, 3, 1, 5, 2],
    "mileage": [120000, 100000, 90000, 70000, 50000, 20000, 80000, 30000],
    "price": [7000, 9000, 11000, 16000, 19000, 32000, 14000, 25000]
})

x = df[["engine_size","horsepower","car_age","mileage"]]
y = df["price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=3,
    random_state=42
)

model.fit(x_train,y_train)
preds = model.predict(x_test)
print("MAE:",mean_squared_error(y_test,preds))
print("R2:",r2_score(y_test,preds))

plt.figure(figsize=(8,5))
sns.kdeplot(y_test,color='red',fill=True,label='actual')
sns.kdeplot(preds,color='blue',fill=True,label='predicted')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.savefig('mylight_plot.png')
