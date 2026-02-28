import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

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
   WHERE label <> 'BENIGN'
   )
   UNION ALL
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
   WHERE label = 'BENIGN'
   ORDER BY RANDOM()
   LIMIT 500000
    );
    """

    df = pd.read_sql(query, engine)

    # Replace inf / -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df.to_parquet(
        "traffic_ml_filtered.parquet",
        engine="pyarrow",
        compression="snappy"
    )

#separating attacks and normal traffic
df["isattack"]=df["label"].apply(lambda x: 0 if x == "BENIGN" else 1)

print(df.info())
print(df.describe())
print(df.shape)
# Splitting the dataset for training and testing 
count_attack = df["isattack"].sum()
count_benign = len(df) - count_attack
print(f"Number of attack samples: {count_attack}")
print(f"Number of benign samples: {count_benign}")
X = df.drop(["label", "isattack"], axis=1)
y = df["isattack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Data Visualization
# Visualize class distribution
class_counts = y.value_counts()
plt.figure(figsize=(6,4))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Class (0 = Benign, 1 = Attack)")
plt.ylabel("Number of Samples")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Visualize distribution of Flow Duration
s=StandardScaler()
scaled=s.fit_transform(X_train[["flow_duration"]])
plt.figure(figsize=(6,4))
plt.hist(scaled, bins=60)
plt.title("Distribution of Flow Duration (Standardized)")
plt.xlabel("Flow Duration (Standardized)")
plt.ylabel("Frequency")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Visualize distribution of Average Packet Size
plt.figure(figsize=(6,4))
plt.hist(X["average_packet_size"], bins=60)
plt.title("Distribution of Average Packet Size")
plt.xlabel("Average Packet Size")
plt.ylabel("Frequency")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()





# Visualize distribution of Flow Bytes per Second by class
benign = X[y == 0]["flow_bytes_per_s"]
attack = X[y == 1]["flow_bytes_per_s"]

plt.figure(figsize=(6,4))
plt.boxplot([benign, attack], labels=["Benign", "Attack"])
plt.title("Flow Bytes per Second by Class")
plt.ylabel("Flow Bytes per Second")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()



# Compute variance for each feature
variances = X.var().sort_values(ascending=False)

top_n = 20
top_variances = variances.head(top_n)

plt.figure(figsize=(7,5))
plt.barh(range(top_n), top_variances.values)
plt.yticks(range(top_n), top_variances.index)
plt.xlabel("Variance")
plt.title("Top 20 Features by Variance")
plt.gca().invert_yaxis()   # Highest variance at top
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()