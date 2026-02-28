from data_loading import X_train, X_test, y_test, y_train
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,f1_score
import joblib
from top40 import TOP40
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
# ============================================
# Model 1: LinearSVC with all features
# ============================================

# 1. Scale features (IMPORTANT!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize LinearSVC
# dual=False is recommended when n_samples > n_features
svm_model = LinearSVC(
    C=1.0,                    # Regularization
    dual=False,               # Use primal formulation (faster for large datasets)
    max_iter=10000,           # Increase if convergence warning appears
    random_state=42,
    verbose=0
)

# 3. Train the model
start_train = time.time()
svm_model.fit(X_train_scaled, y_train)
end_train = time.time() - start_train
# joblib.dump(svm_model, "svm_linear_scaled.pkl")
# svm_model=joblib.load("svm_linear_scaled.pkl")

# 4. Make predictions
start_test = time.time()
y_pred = svm_model.predict(X_test_scaled)
end_test = time.time() - start_test

print("LinearSVC - All Features")
print("Classification Report:")
print(classification_report(y_test, y_pred))
f1_all=(f1_score(y_test,y_pred))


# ============================================
# Model 2: LinearSVC with TOP 40 features
# ============================================

scaler1 = StandardScaler()
X_train_scaled_40 = scaler1.fit_transform(X_train[TOP40])
X_test_scaled_40 = scaler1.transform(X_test[TOP40])

# Initialize LinearSVC for top 40 features
svm_model_40 = LinearSVC(
    C=1.0,
    dual=False,               # dual=False when n_samples > n_features
    max_iter=10000,
    random_state=42,
    verbose=0
)

# Train the model
start_train1 = time.time()
svm_model_40.fit(X_train_scaled_40, y_train)
# joblib.dump(svm_model_40, "svm_linear_scaled_40.pkl")
# svm_model_40=joblib.load("svm_linear_scaled_40.pkl")
end_train1 = time.time() - start_train1  

# Make predictions
start_test1 = time.time()
y_pred1 = svm_model_40.predict(X_test_scaled_40)
end_test1 = time.time() - start_test1

print("LinearSVC - Top 40 Features")
print("Classification Report:")
print(classification_report(y_test, y_pred1))
f1_40=(f1_score(y_test,y_pred1))

results = pd.DataFrame({
    "Models": ["SVM_all", "SVM_top40"],
    "Train_Time": [end_train, end_train1],
    "Test_Time": [end_test, end_test1],
    "F1_Score": [f1_all, f1_40]
})
if not os.path.exists("svm_linear_results.csv"):
    results.to_csv("svm_linear_results.csv", index=False)

print(results)
 
fig , axs = plt.subplots(3,1,figsize=(4,6))
axs[0].bar(results["Models"],results["Train_Time"],width=0.2)
axs[0].set_title("Training Time")
axs[1].bar(results["Models"],results["Test_Time"],width=0.2)
axs[1].set_title("Testing Time")
axs[2].bar(results["Models"],results["F1_Score"],width=0.2)
axs[2].set_title("F1 Score")
plt.tight_layout()
plt.show()  