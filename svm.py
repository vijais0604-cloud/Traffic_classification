import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.kernel_approximation import Nystrom

# Load the dataset
path = "/Users/vijais/Downloads/vs code/traffic_classification/traffic_ml_filtered.parquet"
df = pd.read_parquet(path)

# Drop rows with missing values
df = df.dropna()

# Define rare classes and columns to drop
rare_classes = [
    "Heartbleed",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Web Attack - XSS",
    "Web Attack - Brute Force"
]

columns_to_drop = [
    "rst_flag_count",
    "fin_flag_count",
    "syn_flag_count",
    "urg_flag_count",
    "subflow_fwd_packets"
]

# Drop specified columns
df = df.drop(columns=columns_to_drop)

# Separate features and target
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling for rare classes
smote = SMOTE(sampling_strategy={1: 20000, 8: 5300, 9: 20500, 10: 7200, 5: 8600, 6: 9000, 7: 10000, 3: 13000, 2: 95000}, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)

# Apply the Nystrom method
nystrom = Nystrom(n_components=100, kernel='rbf', gamma='scale', random_state=42)
X_train_nystrom = nystrom.fit_transform(X_train_scaled)
X_val_nystrom = nystrom.transform(X_val_scaled)

# Train the SVM model with the RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_nystrom, y_train_res)

# Evaluate the model
y_pred = svm_model.predict(X_val_nystrom)
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')
weighted_f1 = f1_score(y_val, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Macro F1 Score: {macro_f1}")
print(f"Weighted F1 Score: {weighted_f1}")
