import pandas as pd
import torch
import sys
import numpy as np
import random
sys.path.insert(0, r"D:\Fast Assignments\FYP\backend")

from app.utils.model_loader import ModelInference

print("Loading dataset...")
df = pd.read_csv(r"D:\Fast Assignments\FYP\test SWaT_Dataset_Attack_v0.csv", header=1)

# Extract inputs and labels
sensor_cols = [c for c in df.columns if c.strip() not in ["Timestamp", "Normal/Attack", ""]]

print(f"Dataset shape: {df.shape}")

# Separate normal and attack datasets
# Assuming last column is the label: 0 for normal, 1 for attack
label_col = df.columns[-1]
df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

normal_df = df[df[label_col] == 0.0]
attack_df = df[df[label_col] == 1.0]

print(f"Total Normal rows: {len(normal_df)}")
print(f"Total Attack rows: {len(attack_df)}")

# Sample 100 rows from each
sample_size = min(100, len(attack_df))
normal_samples = normal_df.sample(sample_size, random_state=42)
attack_samples = attack_df.sample(sample_size, random_state=42)

inference = ModelInference(model_path=r"D:\Fast Assignments\FYP\swat_fdai_model_final.pth")

def evaluate_samples(samples, is_attack):
    results = {
        "max_probs": [],
        "mean_probs": [],
        "flagged_nodes": []
    }
    
    for _, row in samples.iterrows():
        # Prepare data dict
        data = {col.strip(): row[col] for col in sensor_cols}
        
        # Predict uses the internal _preprocess_data which NORMALIZES
        pred = inference.predict(data)
        
        nodes = pred["topology"]["nodes"]
        probs = [n["confidence"] for n in nodes]
        
        if probs:
            results["max_probs"].append(max(probs))
            results["mean_probs"].append(sum(probs)/len(probs))
            results["flagged_nodes"].append(len(pred["anomalies"]))
            
    avg_max = np.mean(results["max_probs"])
    avg_mean = np.mean(results["mean_probs"])
    avg_flagged = np.mean(results["flagged_nodes"])
    
    label_type = "ATTACK" if is_attack else "NORMAL"
    print(f"\n--- {label_type} SAMPLES (n={len(samples)}) ---")
    print(f"Mean of max probabilities per row: {avg_max:.4f}")
    print(f"Mean of mean probabilities per row: {avg_mean:.4f}")
    print(f"Average flagged sensors (>0.5): {avg_flagged:.2f} / {len(sensor_cols)}")
    
    return results

print("\nStarting evaluation...")
norm_res = evaluate_samples(normal_samples, is_attack=False)
attk_res = evaluate_samples(attack_samples, is_attack=True)

# Calculate theoretical threshold accuracy: 
# Let's say if mean probability > 0.45 or max probability > 0.85
print("\nThreshold Tests (Graph-level anomaly):")

# Using max prob threshold
threshold_max = 0.90
norm_false_pos = sum(1 for p in norm_res["max_probs"] if p > threshold_max)
attk_true_pos = sum(1 for p in attk_res["max_probs"] if p > threshold_max)

print(f"Rule: Max Node Prob > {threshold_max}")
print(f"  False Positive Rate (Normal flagged as Attack): {norm_false_pos/sample_size*100:.1f}%")
print(f"  True Positive Rate (Attack detected as Attack): {attk_true_pos/sample_size*100:.1f}%")

# Using mean prob threshold
threshold_mean = 0.45
norm_false_pos2 = sum(1 for p in norm_res["mean_probs"] if p > threshold_mean)
attk_true_pos2 = sum(1 for p in attk_res["mean_probs"] if p > threshold_mean)

print(f"\nRule: Mean Node Prob > {threshold_mean}")
print(f"  False Positive Rate (Normal flagged as Attack): {norm_false_pos2/sample_size*100:.1f}%")
print(f"  True Positive Rate (Attack detected as Attack): {attk_true_pos2/sample_size*100:.1f}%")
