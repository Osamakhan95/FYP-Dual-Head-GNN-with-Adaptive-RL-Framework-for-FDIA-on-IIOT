import pandas as pd
import numpy as np
import random
import sys

# Ensure backend modules can be imported
sys.path.insert(0, r"D:\Fast Assignments\FYP\backend")
from app.utils.model_loader import model_inference

def evaluate_model_performance():
    print("Loading SWaT Dataset...")
    df = pd.read_csv(r"D:\Fast Assignments\FYP\test SWaT_Dataset_Attack_v0.csv", header=1)
    
    # Strip column names
    sensor_cols = [c for c in df.columns if c.strip() not in ["Timestamp", "Normal/Attack", ""]]
    
    # Extract label column and ensure it is float
    label_col = df.columns[-1]
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    
    # Group rows by actual class
    normal_df = df[df[label_col] == 0.0]
    attack_df = df[df[label_col] == 1.0]
    
    print(f"Total Normal rows: {len(normal_df)}")
    print(f"Total Attack rows: {len(attack_df)}")
    
    # Sample 100 of each
    np.random.seed(42)
    random.seed(42)
    sample_size = 100
    
    normal_sample = normal_df.sample(sample_size, random_state=42)
    attack_sample = attack_df.sample(sample_size, random_state=42)
    
    print(f"\n=============================================================")
    print(f" EVALUATING TOTAL MODEL PERFORMANCE ({sample_size} Normal / {sample_size} Attack)")
    print(f"=============================================================")
    
    def process_samples(samples, is_ground_truth_attack):
        system_probs_max = []
        system_probs_mean = []
        
        for i, (_, row) in enumerate(samples.iterrows()):
            data = {c.strip(): row[c] for c in sensor_cols}
            res = model_inference.predict(data)
            
            # Use max prob as system anomaly probability (since attack on 1 node = system attack)
            max_prob = res['system_probability']
            system_probs_max.append(max_prob)
            
            # Let's also check the mean probability of all nodes
            probs = [n['confidence'] for n in res['topology']['nodes']]
            system_probs_mean.append(sum(probs)/len(probs))
            
            # Print progress
            if (i+1) % 50 == 0:
                print(f" Processed {i+1}/{sample_size} ...")
                
        return system_probs_max, system_probs_mean

    print("\n[+] Processing 100 Normal Rows (Label = 0)")
    normal_max_probs, normal_mean_probs = process_samples(normal_sample, False)
    
    print("\n[+] Processing 100 Attack Rows (Label = 1)")
    attack_max_probs, attack_mean_probs = process_samples(attack_sample, True)

    # -------------------------------------------------------------
    # CALCULATE METRICS OVER DIFFERENT THRESHOLDS AND LOGICS
    # -------------------------------------------------------------
    
    print("\n=============================================================")
    print(" PERFORMANCE RESULTS & PREDICTION ACCURACY")
    print("=============================================================")
    
    print(f"Statistics:")
    print(f"  Normal Rows -> Average Max Prob: {np.mean(normal_max_probs):.4f}")
    print(f"  Attack Rows -> Average Max Prob: {np.mean(attack_max_probs):.4f}")
    
    # In some predictive models for anomaly detection (like autoencoders/GNNs), 
    # attacks might be identified by an unexpected DROP in probability or confidence,
    # or by extremely high confidence. Let's calculate the accuracy based on standard > 0.5 rules.
    
    # 1. Standard Logic (Max Node Probability > Threshold)
    thresholds = [0.5, 0.7, 0.85, 0.90, 0.95]
    print(f"\n[METHOD A] High Probability = Attack (Max Node Prob > Threshold)")
    for t in thresholds:
        true_negatives = sum(1 for p in normal_max_probs if p <= t)
        false_positives = sum(1 for p in normal_max_probs if p > t)
        
        true_positives = sum(1 for p in attack_max_probs if p > t)
        false_negatives = sum(1 for p in attack_max_probs if p <= t)
        
        accuracy = (true_positives + true_negatives) / 200.0 * 100
        print(f"  Threshold > {t:.2f} | Accuracy: {accuracy:.1f}%  (TPR: {true_positives}%, TNR: {true_negatives}%)")
        
    print(f"\n[METHOD B] Anomaly Deviation from Baseline (Mean Node Prob > Threshold)")
    thresholds_mean = [0.3, 0.4, 0.45, 0.5]
    for t in thresholds_mean:
        true_negatives = sum(1 for p in normal_mean_probs if p <= t)
        false_positives = sum(1 for p in normal_mean_probs if p > t)
        
        true_positives = sum(1 for p in attack_mean_probs if p > t)
        false_negatives = sum(1 for p in attack_mean_probs if p <= t)
        
        accuracy = (true_positives + true_negatives) / 200.0 * 100
        print(f"  Threshold > {t:.2f} | Accuracy: {accuracy:.1f}%  (TPR: {true_positives}%, TNR: {true_negatives}%)")
        
    # Just in case model training inverted the targets (0=Anomaly, 1=Normal)
    print(f"\n[METHOD C] Low Probability = Attack (Inverted Graph Logic)")
    invert_thresholds = [0.2, 0.1, 0.05, 0.01]
    for t in invert_thresholds:
        # For this, we check if minimum probability drops below threshold
        true_neg = sum(1 for p in normal_mean_probs if p > t)
        true_pos = sum(1 for p in attack_mean_probs if p <= t)
        acc = (true_neg + true_pos) / 200.0 * 100
        print(f"  Threshold Mean Prob < {t:.2f} | Accuracy: {acc:.1f}% (TPR: {true_pos}%, TNR: {true_neg}%)")

if __name__ == "__main__":
    evaluate_model_performance()
