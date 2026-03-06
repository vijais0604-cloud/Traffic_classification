import pandas as pd
import matplotlib.pyplot as plt

# Load result dataframes from each model
result_xgb = pd.read_csv("xgb_result.csv")
result_logistic_regression = pd.read_csv("logistic_regression_result.csv")
result_random_forest = pd.read_csv("random_forest_result.csv")

# Concatenate the dataframes
result_combined = pd.concat([result_xgb, result_logistic_regression, result_random_forest], ignore_index=True)

# Create a comparison plot
fig, axs = plt.subplots(len(result_combined.columns), 1, figsize=(6, 8 * len(result_combined.columns)))
for i, column in enumerate(result_combined.columns):
    axs[i].bar(result_combined["Model"], result_combined[column], label=column)
    axs[i].set_title(column)
    axs[i].legend()

plt.tight_layout()
plt.show()
