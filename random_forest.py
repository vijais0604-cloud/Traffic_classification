from top40 import TOP40
import pandas as pd
import joblib
from data_loading import X_test, X_train, y_test,y_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Using only top 40 features for models like Random Forest
X_train_40 = X_train[TOP40]
X_test_40 = X_test[TOP40]

# rf_classifier = RandomForestClassifier(
#     n_estimators=100,        # Number of trees
#     max_depth=None,          # Maximum depth of trees
#     min_samples_split=2,     # Minimum samples to split a node
#     min_samples_leaf=1,      # Minimum samples at leaf node
#     random_state=42,
#     n_jobs=-1                # Use all processors
# )


# Train the model
# start=time.time()
rf_classifier=joblib.load("rf_classifier_40.pkl")
# rf_classifier.fit(X_train_40, y_train)
# joblib.dump(rf_classifier, "rf_classifier_40.pkl")
# Make predictions
y_pred = rf_classifier.predict(X_test_40)
# end=time.time()

# Evaluate the model
# For Classification:
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
print("Random forest classification report")
print(classification_report(y_test, y_pred))

