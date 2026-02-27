from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from top40 import TOP40
from sklearn.metrics import classification_report,f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import joblib
from data_loading import X_test, X_train, y_test,y_train
import time
import matplotlib.pyplot as plt

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
X_test_mi_scaled =scaler1.transform(X_test[TOP40])





# Training the LogisticRegression model with scaled features and loading model
start_train = time.time()
model_scaled = LogisticRegression(max_iter=200)
model_scaled.fit(X_train_scaled, y_train)
end_train = time.time() - start_train
# joblib.dump(model_scaled, "logistic_regression_model_scaled.pkl")

# model_scaled=joblib.load("logistic_regression_model_scaled.pkl")

# Predicting and evaluating the model with scaled features
start_test = time.time()
y_pred_scaled = model_scaled.predict(X_test_scaled)
end_test = time.time() - start_test                                                                                                                                         
print("Logistic Regression Model (Scaled Features)")
print(classification_report(y_test, y_pred_scaled))

f1_all=(f1_score(y_test,y_pred_scaled))

# Training the LogisticRegression model with top40 scaled features and loading model

start_train1 = time.time()
model_scaled_mi = LogisticRegression(max_iter=200)
model_scaled_mi.fit(X_train_mi_scaled, y_train)
# joblib.dump(model_scaled_mi, "logistic_regression_model_mi_scaled.pkl")
end_train1 = time.time() - start_train1

# model_scaled_mi=joblib.load("logistic_regression_model_mi_scaled.pkl")

# Predicting and evaluating the model with scaled features
start_test1 = time.time()
y_pred_scaled_mi = model_scaled_mi.predict(X_test_mi_scaled)
end_test1= time.time() - start_test1
print("Logistic Regression Model with top 40 (Scaled Features) ")
print(classification_report(y_test, y_pred_scaled_mi))
f1_top40=(f1_score(y_test,y_pred_scaled_mi))


results = pd.DataFrame({
    "Feature_Set": ["All Features", "Top40"],
    "Train_Time": [end_train, end_train1],
    "Test_Time": [end_test, end_test1],
    "F1_Score": [f1_all, f1_top40]
})

print(results)


fig , ax = plt.subplots(3,1,figsize=(4,6))
ax[0].bar(results["Feature_Set"],results["Train_Time"],width=0.2)
ax[0].set_title("Training time comparsion")
ax[0].set_ylabel("Time (seconds)")

ax[1].bar(results["Feature_Set"],results["Test_Time"],width=0.2)
ax[1].set_title("Testing time comparsion")
ax[1].set_ylabel("Time (seconds)")

ax[2].bar(results["Feature_Set"],results["F1_Score"],width=0.2)
ax[2].set_title("F1 score comparsion")
ax[2].set_ylabel("F1 score")

plt.tight_layout()
plt.show()