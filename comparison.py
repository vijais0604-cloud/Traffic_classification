import pandas as pd
import matplotlib.pyplot as plt

# Load result dataframes from each model
result_xgb = pd.read_csv("xgb_result.csv")
result_logistic_regression = pd.read_csv("logistic_regression_result.csv")
result_random_forest = pd.read_csv("random_forest_result.csv")

# Concatenate the dataframes
result_combined = pd.concat([result_xgb, result_logistic_regression, result_random_forest], ignore_index=True)

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