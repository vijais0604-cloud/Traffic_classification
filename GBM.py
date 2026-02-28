from top40 import TOP40
import pandas as pd
import joblib
from data_loading import X_test, X_train, y_test, y_train
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import time
import os
xgb_model = XGBClassifier()
xgb_model1 = XGBClassifier()

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores_xgb_all = cross_val_score(
#     xgb_model,
#     X_train,
#     y_train,
#     cv=cv,
#     scoring="f1",
#     n_jobs=1
# )

# print("XGBoost (All Features) CV F1: %.4f ± %.4f" %
#       (scores_xgb_all.mean(), scores_xgb_all.std()))

# scores_xgb_mi = cross_val_score(
#     xgb_model1,
#     X_train[TOP40],
#     y_train,
#     cv=cv,
#     scoring="f1",
#     n_jobs=1
# )

# print("XGBoost (Top40) CV F1: %.4f ± %.4f" %
#       (scores_xgb_mi.mean(), scores_xgb_mi.std()))


start_train = time.time()
xgb_model.fit(X_train, y_train)
# joblib.dump(xgb_model, "xgb_model_all_features.pkl")
# xgb_model = joblib.load("xgb_model_all_features.pkl")
end_train = time.time() - start_train

start_test = time.time()
y_pred_all = xgb_model.predict(X_test)
end_test= time.time() - start_test
print("Classification Report for XGBoost (all features):")
print(classification_report(y_test, y_pred_all))
f1_all = f1_score(y_test, y_pred_all)


start_train1 = time.time()
xgb_model1.fit(X_train[TOP40], y_train)
# joblib.dump(xgb_model1, "xgb_model_top40.pkl")
# xgb_model1 = joblib.load("xgb_model_top40.pkl")
end_train1 = time.time() - start_train1

start_test1 = time.time()
y_pred_mi = xgb_model1.predict(X_test[TOP40])

end_test1 = time.time() - start_test1

print("Classification Report for XGBoost (top40):")
print(classification_report(y_test, y_pred_mi))
f1_top40= f1_score(y_test, y_pred_mi)


results = pd.DataFrame({
    "Models": ["XGB_all", "XGB_top40"],
    "Train_Time": [end_train, end_train1],
    "Test_Time": [end_test, end_test1],
    "F1_Score": [f1_all, f1_top40]
}) 
if  not os.path.exists("xgboost_results.csv"):
    results.to_csv("xgboost_results.csv", index=False)

print(results)

fig, ax = plt.subplots(3, 1, figsize=(4, 4))
ax[0].bar(results["Models"], results["Train_Time"], width=0.2)
ax[0].set_title("Training time comparison")
ax[0].set_ylabel("Time (seconds)")

ax[1].bar(results["Models"], results["Test_Time"], width=0.2)
ax[1].set_title("Testing time comparison")
ax[1].set_ylabel("Time (seconds)")

ax[2].bar(results["Models"], results["F1_Score"], width=0.2)
ax[2].set_title("F1 score comparison")
ax[2].set_ylabel("F1 score")

plt.tight_layout()
plt.show()
