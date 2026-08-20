import pandas as pd
import numpy as np
import sys
import warnings

# Suppress minor warnings
warnings.filterwarnings('ignore')

# Ensure backend modules can be imported
sys.path.insert(0, r"D:\Fast Assignments\FYP\backend")
from app.utils.model_loader import model_inference

def evaluate_total_performance():
    print("Loading SWaT Dataset...")
    df = pd.read_csv(r"D:\Fast Assignments\FYP\test SWaT_Dataset_Attack_v0.csv", header=1)
    
    # Clean column matching
    sensor_cols = [c for c in df.columns if c.strip() not in ["Timestamp", "Normal/Attack", ""]]
    
    # Cast label exactly
    label_col = df.columns[-1]
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    
    # Split
    normals = df[df[label_col] == 0.0]
    attacks = df[df[label_col] == 1.0]
    
    # Sample exactly 100 random rows from each group
    sample_size = 100
    normal_sample = normals.sample(sample_size, random_state=123)
    attack_sample = attacks.sample(sample_size, random_state=123)
    
    print("\n--------------------------------------------------------------")
    print(f" TESTING: {sample_size} NORMAL ROWS VS {sample_size} ATTACK ROWS")
    print("--------------------------------------------------------------")
    
    def process_rows(samples, name):
        correct = 0
        total = len(samples)
        
        for _, row in samples.iterrows():
            data = {c.strip(): row[c] for c in sensor_cols}
            res = model_inference.predict(data)
            
            # Use the newly added sigmoid mapped system probability logic
            predicted_attack = res['is_attack']
            
            # Tally correct flags
            if name == "Normal":
                if not predicted_attack: correct += 1
            else:
                if predicted_attack: correct += 1
                
        percentage = (correct / total) * 100
        print(f"[{name}] Predicted Correctly: {correct}/{total} -> {percentage:.1f}% Accuracy")
        return correct, total

    norm_c, norm_t = process_rows(normal_sample, "Normal")
    atk_c, atk_t = process_rows(attack_sample, "Attack")

    print("\n==============================================================")
    print("                 FINAL INFERENCE SUMMARY")
    print("==============================================================")
    total_correct = norm_c + atk_c
    total_rows = norm_t + atk_t
    print(f"Total Model Evaluated Accuracy: {(total_correct / total_rows) * 100:.1f}%\n")
    print("Details:")
    print(f"- True Positive Rate (Detected Attacks): {(atk_c / atk_t) * 100:.1f}%")
    print(f"- True Negative Rate (Detected Normal):  {(norm_c / norm_t) * 100:.1f}%")
    print("==============================================================")

if __name__ == "__main__":
    evaluate_total_performance()
