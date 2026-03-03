import joblib
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

#checking for the dataset in the given path, if not found creating the dataset from database by filtering 900k rows from 2.8M rows
path="/Users/vijais/Downloads/vs code/traffic_classification/traffic_ml_filtered.parquet"
if os.path.exists(path):
    df = pd.read_parquet("/Users/vijais/Downloads/vs code/traffic_classification/traffic_ml_filtered.parquet")
else:
    engine = create_engine(
        "postgresql+psycopg2://vijais@localhost:5432/traffic_ml_db"
    )

    query = """
   (
   SELECT
   flow_duration,
   total_fwd_packets,
   total_backward_packets,
   total_length_fwd_packets,
   total_length_bwd_packets,
   fwd_packet_length_max,
   fwd_packet_length_min,
   fwd_packet_length_mean,
   fwd_packet_length_std,
   bwd_packet_length_max,
   bwd_packet_length_min,
   bwd_packet_length_mean,
   bwd_packet_length_std,
   flow_bytes_per_s,
   flow_packets_per_s,
   flow_iat_mean,
   flow_iat_std,
   flow_iat_max,
   flow_iat_min,
   fwd_iat_total,
   fwd_iat_mean,
   fwd_iat_std,
   fwd_iat_max,
   fwd_iat_min,
   bwd_iat_total,
   bwd_iat_mean,
   bwd_iat_std,
   bwd_iat_max,
   bwd_iat_min,
   min_packet_length,
   max_packet_length,
   packet_length_mean,
   packet_length_std,
   packet_length_variance,
   fin_flag_count,
   syn_flag_count,
   rst_flag_count,
   psh_flag_count,
   ack_flag_count,
   urg_flag_count,
   down_up_ratio,
   average_packet_size,
   avg_fwd_segment_size,
   avg_bwd_segment_size,
   subflow_fwd_packets,
   subflow_fwd_bytes,
   subflow_bwd_packets,
   subflow_bwd_bytes,
   init_win_bytes_forward,
   init_win_bytes_backward,
   act_data_pkt_fwd,
   min_seg_size_forward,
   active_mean,
   active_std,
   active_max,
   active_min,
   idle_mean,
   idle_std,
   idle_max,
   idle_min,
   label
   FROM traffic_data
   )
 """

    df = pd.read_sql(query, engine)
    print("Before:", df.shape)
    df = df.drop_duplicates()
    print("After:", df.shape)

    df.to_parquet(
        "traffic_ml_filtered.parquet",
        engine="pyarrow",
        compression="snappy"
    )

df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print(df.isnull().sum().sort_values(ascending=False).head(20))
# print("Before:", df.shape)
df = df.dropna()
# print("After:", df.shape)
df["label"] = df["label"].str.strip()
df["label"] = df["label"].str.replace("�", "-", regex=False)
df["label"] = df["label"].str.replace("  ", " ", regex=False)

rare_classes = [
    "Heartbleed",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Web Attack - XSS",
    "Web Attack - Brute Force"
]

df["label"] = df["label"].apply(
    lambda x: "Rare_Attack" if x in rare_classes else x
)

# print(df["label"].value_counts())
# print(df["label"].unique())

# variances = df.drop(columns=["label"]).var().sort_values()
# print(variances.head(15))

# corr_matrix = df.drop(columns=["label"]).corr().abs()
# upper = corr_matrix.where(
#     np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
# )
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# print("Highly correlated features:", to_drop)

columns_to_drop = [

    # Near zero variance
    "rst_flag_count",
    "fin_flag_count",
    "syn_flag_count",
    "urg_flag_count",

    # Redundant subflow
    "subflow_fwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_packets",
    "subflow_bwd_bytes",

    # Packet length redundancy
    "fwd_packet_length_std",
    "bwd_packet_length_std",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "average_packet_size",

    # Highly correlated counts
    "total_backward_packets",
    "total_length_bwd_packets",

    # IAT redundancy
    "fwd_iat_max",

    # Idle redundancy
    "idle_max",
    "idle_min"
]
df = df.drop(columns=columns_to_drop)
# print(df.shape)



if not os.path.exists("label_encoder.pkl"):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    joblib.dump(le, "label_encoder.pkl")
else:
    le = joblib.load("label_encoder.pkl")

df["label_encoded"] = le.transform(df["label"])

# print(df[["label", "label_encoded"]].head(10))



X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]



from sklearn.model_selection import train_test_split
X_t, X_temp, y_t, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,          # 30% goes to temp
    random_state=42,
    stratify=y               # CRITICAL for imbalance
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,          # Half of 30% → 15% each
    random_state=42,
    stratify=y_temp
)


# Combine X_train and y_train temporarily
train_df = X_t.copy()
train_df["label"] = y_t

# Separate benign and attack
benign_df = train_df[train_df["label"] == le.transform(["BENIGN"])[0]]
attack_df = train_df[train_df["label"] != le.transform(["BENIGN"])[0]]

# Undersample benign
benign_sampled = benign_df.sample(n=450000, random_state=42)

# Combine back
train_balanced = pd.concat([benign_sampled, attack_df])

# Shuffle
train_balanced = train_balanced.sample(frac=1, random_state=42)

# Separate again
X_train = train_balanced.drop(columns=["label"])
y_train = train_balanced["label"]