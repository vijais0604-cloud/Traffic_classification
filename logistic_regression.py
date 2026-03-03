from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
from data_loading import X_train,X_val, y_train, y_val, X_test, y_test
import pandas as pd
import joblib
import os
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# print(pd.Series(y_train).value_counts())

# from imblearn.over_sampling import SMOTE
# smote = SMOTE(
#     sampling_strategy={1:20000,8:5300,9:20500,10:7200,5:8600,6:9000,7:10000,3:13000,2:95000},  # oversample all minority classe
#     random_state=42
# )
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
# import pandas as pd
# print(pd.Series(y_train_smote).value_counts())


# start_time = time.time()
# log_model = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     C=0.5
# )
# log_model.fit(X_train_smote, y_train_smote)
# training_time = time.time() - start_time
# print("Training Time:", training_time)
# joblib.dump(log_model, "logistic_regression_model.pkl")

# start_time1 = time.time()
# y_pred = log_model.predict(X_val_scaled)
# testing_time= time.time() - start_time1
# print("Testing Time:", testing_time)



# accuracy = accuracy_score(y_val, y_pred)
# macro_f1 = f1_score(y_val, y_pred, average='macro')
# weighted_f1 = f1_score(y_val, y_pred, average='weighted')

# print("Accuracy:", accuracy)
# print("Macro F1:", macro_f1)
# print("Weighted F1:", weighted_f1)
# print("\nClassification Report:\n")
# print(classification_report(y_val, y_pred))


log_model1=joblib.load("logistic_regression_model.pkl")
y_test_pred = log_model1.predict(X_test_scaled)
accuracy_test = accuracy_score(y_test, y_test_pred)
macro_f1_test = f1_score(y_test, y_test_pred, average='macro')
weighted_f1_test = f1_score(y_test, y_test_pred, average='weighted')      
print(classification_report(y_test, y_test_pred))
print("Accuracy:", accuracy_test)
print("Macro F1:", macro_f1_test)
print("Weighted F1:", weighted_f1_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))



