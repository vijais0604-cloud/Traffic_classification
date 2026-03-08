import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.kernel_approximation import Nystrom
from data_loading import X_train, X_val, y_train, y_val
from logistic_regression import evaluate_model

# Apply the Nystrom method
nystrom = Nystrom(n_components=100, kernel='rbf', gamma='scale', random_state=42)
X_train_nystrom = nystrom.fit_transform(X_train)
X_val_nystrom = nystrom.transform(X_val)

# Train the SVM model with the RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_nystrom, y_train)

# Evaluate the model
result = evaluate_model(svm_model, X_val_nystrom, y_val)

# Print the results
print(result)
