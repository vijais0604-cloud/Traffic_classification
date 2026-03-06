import pandas as pd
import matplotlib.pyplot as plt

# Load result dataframes from each model
result_xgb = pd.read_csv("xgb_result.csv")
result_logistic_regression = pd.read_csv("logistic_regression_result.csv")
result_random_forest = pd.read_csv("random_forest_result.csv")

# Concatenate the dataframes
result_combined = pd.concat([result_xgb, result_logistic_regression, result_random_forest], ignore_index=True)

# Create a comparison plot
figs = []
for column in result_combined.columns:
    fig, axs = plt.subplots()
    axs.bar(result_combined["Model"], result_combined[column], label=column)
    axs.set_title(column)
    axs.legend()
    figs.append(fig)

for i, fig in enumerate(figs):
    fig.tight_layout()
    fig.show()
