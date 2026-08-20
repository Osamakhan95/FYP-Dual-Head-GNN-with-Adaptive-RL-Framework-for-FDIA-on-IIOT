import pandas as pd
import sys
import json
sys.path.insert(0, r"D:\Fast Assignments\FYP\backend")
from app.utils.model_loader import model_inference

def run_test():
    print("Loading dataset...")
    df = pd.read_csv(r"D:\Fast Assignments\FYP\test SWaT_Dataset_Attack_v0.csv", header=1)
    sensor_cols = [c for c in df.columns if c.strip() not in ["Timestamp", "Normal/Attack", ""]]
    
    # Get one standard normal row
    # Index 1000 is Normal
    normal_row = df.iloc[1000]
    normal_data = {c.strip(): normal_row[c] for c in sensor_cols}
    
    # Get one confirmed attack row
    # Index 445190 is Attack
    attack_row = df.iloc[445190]
    attack_data = {c.strip(): attack_row[c] for c in sensor_cols}
    
    print("\n" + "="*50)
    print("TESTING NORMAL ROW (INDEX 1000)")
    print("="*50)
    res_normal = model_inference.predict(normal_data)
    nodes_normal = res_normal.get("topology", {}).get("nodes", [])
    probs_normal = [n["confidence"] for n in nodes_normal]
    
    print(f"Total Anomalies flagged (>0.5): {len(res_normal['anomalies'])} / 51")
    print(f"Mean Node Probability: {sum(probs_normal)/51:.4f}")
    if probs_normal:
        print(f"Max Node Probability: {max(probs_normal):.4f}")
    
    print("\n" + "="*50)
    print("TESTING ATTACK ROW (INDEX 445190)")
    print("="*50)
    res_attack = model_inference.predict(attack_data)
    nodes_attack = res_attack.get("topology", {}).get("nodes", [])
    probs_attack = [n["confidence"] for n in nodes_attack]
    
    print(f"Total Anomalies flagged (>0.5): {len(res_attack['anomalies'])} / 51")
    print(f"Mean Node Probability: {sum(probs_attack)/51:.4f}")
    if probs_attack:
        print(f"Max Node Probability: {max(probs_attack):.4f}")
        
    print("\nDetailed Attack Row Anomalies:")
    for a in res_attack['anomalies']:
        print(f" - {a['node_id']}: {a['confidence']:.4f} ({a['severity']})")

if __name__ == "__main__":
    run_test()
