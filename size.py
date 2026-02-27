
import os

# Get file size
size_bytes = os.path.getsize("xgb_model_all_features.pkl")
size_mb = size_bytes / (1024 * 1024)

size_bytes1 = os.path.getsize("xgb_model_top40.pkl")
size_mb1 = size_bytes1 / (1024 * 1024)

print("XGBoost Model size (all features): %.2f MB" % size_mb)
print("XGBoost Model size (top40 features): %.2f MB" % size_mb1) 

sb_rf= os.path.getsize("rf_classifier.pkl") / (1024 * 1024)
sb_rf_40= os.path.getsize("rf_classifier_40.pkl") / (1024 * 1024)
print("Random Forest Model size (all features): %.2f MB" % sb_rf)
print("Random Forest Model size (top40 features): %.2f MB" % sb_rf_40)

size_lr_all = os.path.getsize("logistic_regression_model_scaled.pkl") / (1024 * 1024)
size_lr_mi = os.path.getsize("logistic_regression_model_mi_scaled.pkl") / (1024 * 1024)
print("Logistic Regression Model size (all features): %.2f MB" % size_lr_all)
print("Logistic Regression Model size (top40 features): %.2f MB" % size_lr_mi)


size_svm_rbf_all = os.path.getsize("svm_rbf_approx_scaled_40.pkl") / (1024 * 1024)
size_svm_rbf_40 = os.path.getsize("svm_rbf_approx_scaled.pkl") / (1024 * 1024)
print("SVM RBF Model size (all features): %.2f MB" % size_svm_rbf_all)
print("SVM RBF Model size (top40 features): %.2f MB" % size_svm_rbf_40)