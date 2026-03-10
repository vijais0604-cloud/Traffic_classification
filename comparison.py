import pandas as pd
import matplotlib.pyplot as plt
import os

# Load result dataframes from each model
result_logistic_regression = pd.read_csv("logistic_regression_result.csv")
result_random_forest = pd.read_csv("random_forest_result.csv")
result_lg = pd.read_csv("lgb_result.csv")
result_xgb = pd.read_csv("xgb_result_tuned.csv")

lr=os.path.getsize("logistic_regression_model.pkl")/(1024 * 1024)
rf=os.path.getsize("random_forest_model.pkl")/(1024 * 1024)
lg=os.path.getsize("lightgbm_model.pkl")/(1024 * 1024)
xgb=os.path.getsize("xgb_model_tuned.pkl")/(1024 * 1024)
xgb_base=os.path.getsize("xgb_model.pkl")/(1024 * 1024)
# Add size of model column
# Concatenate the dataframes
result_combined = pd.concat([result_xgb, result_logistic_regression, result_random_forest, result_lg], ignore_index=True)
result_combined["Size of model (MB)"]=[xgb_base,xgb,lr,rf,lg]
print(result_combined)

plt.bar(result_combined["Model"], result_combined["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.show()

plt.bar(result_combined["Model"], result_combined["Macro_f1"])
plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.title("Macro F1 Comparison")
plt.show()  

plt.bar(result_combined["Model"], result_combined["F1 score"])
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison")
plt.show()

plt.bar(result_combined["Model"], result_combined["Training time"])
plt.xlabel("Model")
plt.ylabel("Training Time")
plt.title("Training Time Comparison")
plt.show()

plt.bar(result_combined["Model"], result_combined["Testing time"])
plt.xlabel("Model")
plt.ylabel("Testing Time")
plt.title("Testing Time Comparison")
plt.show()

plt.bar(result_combined["Model"], result_combined["Size of model (MB)"])
plt.xlabel("Model")
plt.ylabel("Size of model (MB)")
plt.title("Size of model (MB) Comparison")
plt.show()