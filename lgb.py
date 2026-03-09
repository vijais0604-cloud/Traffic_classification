import time
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_loading import X_train, y_train, X_val, y_val

# Model
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    class_weight='balanced',
    random_state=42
)

# Training
start_train = time.time()
lgb_model.fit(X_train, y_train)
end_train = time.time()

joblib.dump(lgb_model, "lightgbm_model.pkl")

# Testing
start_test = time.time()
y_pred = lgb_model.predict(X_val)
end_test = time.time()

# Metrics
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')
weighted_f1 = f1_score(y_val, y_pred, average='weighted')

print("Training Time:", end_train - start_train)
print("Testing Time:", end_test - start_test)
print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))