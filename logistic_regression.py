from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from top40 import TOP40
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import joblib
from data_loading import X_test, X_train, y_test,y_train
import time

# Creating a file to store the top40 columns if it doesn't exist
path1 = "/Users/vijais/Documents/vs code/traffic_classification/top40.py"
if not os.path.exists(path1):
    # Finding the mutual correlation between the columns
    mi_scores = mutual_info_classif(X_train, y_train)
    # Getting the top 40 columns with highest mutual information
    mi_series = pd.Series(mi_scores, index=X_train.columns)
    mi_top40 = mi_series.sort_values(ascending=False).head(40).index.tolist()
    with open("top40.py", "w") as f:
        f.write("TOP40 = " + str(mi_top40))


# Scaling the features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler1 = StandardScaler()
X_train_mi_scaled = scaler1.fit_transform(X_train[TOP40])
X_test_mi_scaled =scaler1.fit_transform(X_test[TOP40])





# Training the LogisticRegression model with scaled features and loading model

# model_scaled = LogisticRegression(max_iter=200)
# model_scaled.fit(X_train_scaled, y_train)
# joblib.dump(model_scaled, "logistic_regression_model_scaled.pkl")
# start_time1 = time.time()

model_scaled=joblib.load("logistic_regression_model_scaled.pkl")

# Predicting and evaluating the model with scaled features
y_pred_scaled = model_scaled.predict(X_test_scaled)
# end_time1 = time.time()                                                                                                                                               
print("Logistic Regression Model (Scaled Features)")
print(classification_report(y_test, y_pred_scaled))



# Training the LogisticRegression model with top40 scaled features and loading model


# model_scaled_mi = LogisticRegression(max_iter=200)
# model_scaled_mi.fit(X_train_mi_scaled, y_train)
# joblib.dump(model_scaled_mi, "logistic_regression_model_mi_scaled.pkl")
# start_time = time.time()

model_scaled_mi=joblib.load("logistic_regression_model_mi_scaled.pkl")

# Predicting and evaluating the model with scaled features
y_pred_scaled_mi = model_scaled_mi.predict(X_test_mi_scaled)
# end_time = time.time()
print("Logistic Regression Model with top 40 (Scaled Features) ")
print(classification_report(y_test, y_pred_scaled_mi))
