"""Train and evaluate a baseline delivery-time regression model."""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("simulated_logistics_shipments.csv")
X = df[["distance_km","weight_kg","traffic_level","weather","vehicle_type","priority","zone"]]
y = df["delivery_hours"]
num = ["distance_km","weight_kg"]
cat = ["traffic_level","weather","vehicle_type","priority","zone"]
pre = ColumnTransformer([("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num), ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)])
model = Pipeline([("preprocess",pre),("regressor",RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("MAE hours:", round(mean_absolute_error(y_test,pred),3))
print("RMSE hours:", round(np.sqrt(mean_squared_error(y_test,pred)),3))
print("R-squared:", round(r2_score(y_test,pred),3))
cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print("5-fold CV MAE:", round(-cv.mean(),3), "+/-", round(cv.std(),3))
# Predictions should support human decisions, not automatically dispatch vehicles.
