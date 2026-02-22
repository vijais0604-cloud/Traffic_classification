from top40 import TOP40
import pandas as pd
import joblib
from data_loading import X_test, X_train, y_test,y_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import time
import matplotlib.pyplot as plt

# Using only top 40 features for models like Random Forest
X_train_40 = X_train[TOP40]
X_test_40 = X_test[TOP40]

rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth of trees
    min_samples_split=2,     # Minimum samples to split a node
    min_samples_leaf=1,      # Minimum samples at leaf node
    random_state=42,
    n_jobs=-1                # Use all processors
)

rf_classifier1 = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth of trees
    min_samples_split=2,     # Minimum samples to split a node
    min_samples_leaf=1,      # Minimum samples at leaf node
    random_state=42,
    n_jobs=-1                # Use all processors
)

start_train=time.time()
# rf_classifier=joblib.load("rf_classifier_40.pkl")
rf_classifier.fit(X_train_40, y_train)
end_train=time.time() - start_train
# joblib.dump(rf_classifier, "rf_classifier_40.pkl")
# Make predictions
start_test=time.time()
y_pred = rf_classifier.predict(X_test_40)
end_test=time.time() - start_train
# Evaluate the model
# For Classification:
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
print("Random forest classification report")
print(classification_report(y_test, y_pred))
f1_all=f1_score(y_test,y_pred)


# Train the model
start_train1=time.time()
# rf_classifier=joblib.load("rf_classifier_40.pkl")
rf_classifier1.fit(X_train_40, y_train)
end_train1=time.time() - start_train1
# joblib.dump(rf_classifier, "rf_classifier_40.pkl")
# Make predictions
start_test1=time.time()
y_pred1 = rf_classifier1.predict(X_test_40)
end_test1=time.time() - start_train1
# Evaluate the model
# For Classification:
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
print("Random forest classification report")
print(classification_report(y_test, y_pred1))
f1_top40=f1_score(y_test,y_pred1)

results = pd.DataFrame({
    "Feature_Set": ["All Features", "Top40"],
    "Train_Time": [end_train, end_train1],
    "Test_Time": [end_test, end_test1],
    "F1_Score": [f1_all, f1_top40]
})
print(results)

fig , ax = plt.subplots(3,1,figsize=(4,4))
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