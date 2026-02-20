import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
from sklearn.model_selection import train_test_split


#checking for the dataset in the given path, if not found creating the dataset from database by filtering 900k rows from 2.8M rows
path="/Users/vijais/Documents/vs code/traffic_classification/traffic_ml_filtered.parquet"
if os.path.exists(path):
    df = pd.read_parquet("/Users/vijais/Documents/vs code/traffic_classification/traffic_ml_filtered.parquet")
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





# Splitting the dataset for training and testing 
X = df.drop(["label", "isattack"], axis=1)
y = df["isattack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
