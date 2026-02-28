from data_loading import X_train, X_test, y_test, y_train
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,f1_score
import joblib
from top40 import TOP40
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
# ============================================
# Model 1: Nystroem + LinearSVC (RBF approximation) - All features
# ============================================

# Build pipeline with RBF approximation
model_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf_feature", Nystroem(
        kernel="rbf",
        gamma=0.1,              # Tune this (similar to RBF SVM gamma)
        n_components=500,       # Higher = better approximation (300-1000)
        random_state=42
    )),
    ("svm", LinearSVC(
        C=1.0,
        dual=True,              # dual=True for transformed space
        max_iter=10000,
        random_state=42
    ))
])

# Train
start_train = time.time()
model_rbf.fit(X_train, y_train)
end_train = time.time() - start_train
# joblib.dump(model_rbf, "svm_rbf_approx_scaled.pkl")
# model_rbf=joblib.load("svm_rbf_approx_scaled.pkl")
# Predict
start_test = time.time()
y_pred = model_rbf.predict(X_test)
end_test = time.time() - start_test

print("=" * 60)
print("Nystroem RBF Approximation - All Features")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
f1_all=(f1_score(y_test,y_pred))

# ============================================
# Model 2: Nystroem + LinearSVC - Top 40 features
# ============================================

model_rbf_40 = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf_feature", Nystroem(
        kernel="rbf",
        gamma=0.1,
        n_components=500,
        random_state=42
    )),
    ("svm", LinearSVC(
        C=1.0,
        dual=True,
        max_iter=10000,
        random_state=42
    ))
])

start_train1 = time.time()
model_rbf_40.fit(X_train[TOP40], y_train)
end_train1 = time.time() - start_train1
# joblib.dump(model_rbf_40, "svm_rbf_approx_scaled_40.pkl")
# model_rbf_40=joblib.load("svm_rbf_approx_scaled_40.pkl")
start_test1 = time.time()
y_pred1 = model_rbf_40.predict(X_test[TOP40])
end_test1 = time.time() - start_test1

print("\n" + "=" * 60)
print("Nystroem RBF Approximation - Top 40 Features")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred1))
f1_40=(f1_score(y_test,y_pred1))


results = pd.DataFrame({
    "Models": ["RBF_all", "RBF_top40"],
    "Train_Time": [end_train, end_train1],
    "Test_Time": [end_test, end_test1],
    "F1_Score": [f1_all, f1_40]
})
if not os.path.exists("svm_rbf_results.csv"):
    results.to_csv("svm_rbf_results.csv", index=False)
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