import pandas as pd
import os
overall_df = pd.DataFrame({"Models":[],
                           "Train_Time":[],
                           "Test_Time":[],
                           "F1_Score":[]})
lr_results = pd.read_csv("logistic_regression_results.csv")
svm_linear_results = pd.read_csv("svm_linear_results.csv")
svm_rbf_results = pd.read_csv("svm_rbf_results.csv")
rf_results = pd.read_csv("random_forest_results.csv")
gbm_results = pd.read_csv("xgboost_results.csv")
overall_df = pd.concat([overall_df, lr_results], ignore_index=True)
overall_df = pd.concat([overall_df, svm_linear_results], ignore_index=True)
overall_df = pd.concat([overall_df, svm_rbf_results], ignore_index=True)
overall_df = pd.concat([overall_df, rf_results], ignore_index=True)
overall_df = pd.concat([overall_df, gbm_results], ignore_index=True)
print(overall_df)
overall_df.to_csv("overall_results.csv", index=False)