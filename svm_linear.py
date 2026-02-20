from data_loading import X_train, X_test, y_test, y_train
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from top40 import TOP40
import time

# ============================================
# Model 1: LinearSVC with all features
# ============================================

# 1. Scale features (IMPORTANT!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize LinearSVC
# dual=False is recommended when n_samples > n_features
# svm_model = LinearSVC(
#     C=1.0,                    # Regularization
#     dual=False,               # Use primal formulation (faster for large datasets)
#     max_iter=10000,           # Increase if convergence warning appears
#     random_state=42,
#     verbose=0
# )

# 3. Train the model
start = time.time()
# svm_model.fit(X_train_scaled, y_train)
# joblib.dump(svm_model, "svm_linear_scaled.pkl")
svm_model=joblib.load("svm_linear_scaled.pkl")

# 4. Make predictions
y_pred = svm_model.predict(X_test_scaled)
end = time.time()

print("LinearSVC - All Features")
print(f"Training time: {end - start} seconds")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# Model 2: LinearSVC with TOP 40 features
# ============================================

scaler1 = StandardScaler()
X_train_scaled_40 = scaler1.fit_transform(X_train[TOP40])
X_test_scaled_40 = scaler1.transform(X_test[TOP40])

# Initialize LinearSVC for top 40 features
# svm_model_40 = LinearSVC(
#     C=1.0,
#     dual=False,               # dual=False when n_samples > n_features
#     max_iter=10000,
#     random_state=42,
#     verbose=0
# )

# Train the model
start1 = time.time()
# svm_model_40.fit(X_train_scaled_40, y_train)
# joblib.dump(svm_model_40, "svm_linear_scaled_40.pkl")
svm_model_40=joblib.load("svm_linear_scaled_40.pkl")
# Make predictions
y_pred1 = svm_model_40.predict(X_test_scaled_40)
end1 = time.time()

print("LinearSVC - Top 40 Features")
print(f"Training time: {end1 - start1} seconds")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred1))