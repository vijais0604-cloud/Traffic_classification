from data_loading import X_train, X_test, y_test, y_train
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from top40 import TOP40
import time

# ============================================
# Model 1: Nystroem + LinearSVC (RBF approximation) - All features
# ============================================

# Build pipeline with RBF approximation
# model_rbf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("rbf_feature", Nystroem(
#         kernel="rbf",
#         gamma=0.1,              # Tune this (similar to RBF SVM gamma)
#         n_components=500,       # Higher = better approximation (300-1000)
#         random_state=42
#     )),
#     ("svm", LinearSVC(
#         C=1.0,
#         dual=True,              # dual=True for transformed space
#         max_iter=10000,
#         random_state=42
#     ))
# ])

# Train
start = time.time()
# model_rbf.fit(X_train, y_train)
# joblib.dump(model_rbf, "svm_rbf_approx_scaled.pkl")
model_rbf=joblib.load("svm_rbf_approx_scaled.pkl")
# Predict
y_pred = model_rbf.predict(X_test)
end = time.time()

print("=" * 60)
print("Nystroem RBF Approximation - All Features")
print("=" * 60)
print(f"Training time: {end - start:.2f} seconds")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# Model 2: Nystroem + LinearSVC - Top 40 features
# ============================================

# model_rbf_40 = Pipeline([
#     ("scaler", StandardScaler()),
#     ("rbf_feature", Nystroem(
#         kernel="rbf",
#         gamma=0.1,
#         n_components=500,
#         random_state=42
#     )),
#     ("svm", LinearSVC(
#         C=1.0,
#         dual=True,
#         max_iter=10000,
#         random_state=42
#     ))
# ])

start1 = time.time()
# model_rbf_40.fit(X_train[TOP40], y_train)
# joblib.dump(model_rbf_40, "svm_rbf_approx_scaled_40.pkl")
model_rbf_40=joblib.load("svm_rbf_approx_scaled_40.pkl")
y_pred1 = model_rbf_40.predict(X_test[TOP40])
end1 = time.time()

print("\n" + "=" * 60)
print("Nystroem RBF Approximation - Top 40 Features")
print("=" * 60)
print(f"Training time: {end1 - start1:.2f} seconds")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred1))