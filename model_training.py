import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from madlan_data_prep import prepare_data


df = pd.read_excel('output_all_students_Train_v10.xlsx')

data = prepare_data(df)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Categorical columns for one-hot encoding
categorical_columns = ["City", "type","city_area", "condition", "furniture", "entrance_date"]

# Numerical columns for standardization
numerical_columns = ['Area','floor']

# Binary columns
binary_columns = ['hasParking','hasBalcony', 'hasMamad']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns),
        ('bin', 'passthrough', binary_columns)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elasticnet', ElasticNet(l1_ratio=0.99, alpha=0.1))
])
final_model = TransformedTargetRegressor(pipeline)

cv_scores = cross_val_score(final_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
mae_scores = cross_val_score(final_model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(final_model, X_train, y_train, cv=10, scoring='r2')
rmse_scores = np.sqrt(mse_scores)


n_train = len(X_train)
k_train = X_train.shape[1]  # Number of predictors (columns)
adj_r2_scores = 1 - ((1 - r2_scores) * (n_train - 1)) / (n_train - k_train - 1)

print("Mean Squared Error (MSE):", mse_scores.mean())
print("Mean Absolute Error (MAE):", mae_scores.mean())
print("Root Mean Squared Error (RMSE):", rmse_scores.mean())
print("R-squared (R^2):", r2_scores.mean())
print("Adjusted R-squared:", adj_r2_scores.mean())


final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

n_test = len(X_test)
k_test = X_test.shape[1]  # Number of predictors (columns)
adj_r2 = 1 - ((1 - r2) * (n_test - 1)) / (n_test - k_test - 1)

print("\n\nTest Set Performance Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse.mean())
print("R-squared (R^2):", r2)
print("Adjusted R-squared:", adj_r2)

pickle.dump(final_model, open("trained_model.pkl","wb"))