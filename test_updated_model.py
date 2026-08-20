"""
Test the updated model_loader.py with min-max normalization.
Uses raw SWaT sensor values — normalization happens inside model_loader.
"""
import sys
sys.path.insert(0, r"D:\Fast Assignments\FYP\backend")

from app.utils.model_loader import ModelInference

# ---- Row 100 from SWaT dataset (Normal row, label=0) ----
normal_row = {
    'FIT101': 2.3579, 'LIT101': 527.7558, 'MV101': 2.0, 'P101': 2.0, 'P102': 1.0,
    'AIT201': 199.6956, 'AIT202': 7.8658, 'AIT203': 339.4104, 'FIT201': 2.358, 'MV201': 2.0,
    'P201': 2.0, 'P202': 1.0, 'P203': 1.0, 'P204': 1.0, 'P205': 2.0, 'P206': 1.0,
    'DPIT301': 19.2634, 'FIT301': 1.1423, 'LIT301': 740.6754, 'MV301': 2.0, 'MV302': 2.0,
    'MV303': 2.0, 'MV304': 2.0, 'P301': 2.0, 'P302': 2.0,
    'AIT401': 183.4506, 'AIT402': 183.5949, 'FIT401': 0.5973, 'LIT401': 461.4395,
    'P401': 2.0, 'P402': 1.0, 'P403': 1.0, 'P404': 1.0, 'UV401': 2.0,
    'AIT501': 6.8072, 'AIT502': 13.2996, 'AIT503': 329.3618, 'AIT504': 8.9204,
    'FIT501': 1.1455, 'FIT502': 0.5992, 'FIT503': 0.3473, 'FIT504': 0.1966,
    'P501': 2.0, 'P502': 1.0, 'PIT501': 263.2085, 'PIT502': 1.4937, 'PIT503': 243.2236,
    'FIT601': 0.0, 'P601': 1.0, 'P602': 1.0, 'P603': 1.0,
}

# ---- Row 400000 from SWaT dataset (Attack row, label=1) ----
attack_row = {
    'FIT101': 0.0, 'LIT101': 789.2946, 'MV101': 0.0, 'P101': 2.0, 'P102': 1.0,
    'AIT201': 212.0684, 'AIT202': 7.6884, 'AIT203': 323.1455, 'FIT201': 2.4107, 'MV201': 0.0,
    'P201': 2.0, 'P202': 2.0, 'P203': 2.0, 'P204': 2.0, 'P205': 2.0, 'P206': 2.0,
    'DPIT301': 0.0, 'FIT301': 0.0, 'LIT301': 535.5155, 'MV301': 2.0, 'MV302': 2.0,
    'MV303': 0.0, 'MV304': 2.0, 'P301': 1.0, 'P302': 1.0,
    'AIT401': 240.8048, 'AIT402': 240.7277, 'FIT401': 0.0, 'LIT401': 375.2327,
    'P401': 1.0, 'P402': 1.0, 'P403': 1.0, 'P404': 1.0, 'UV401': 1.0,
    'AIT501': 9.9373, 'AIT502': 15.2825, 'AIT503': 391.2506, 'AIT504': 16.6283,
    'FIT501': 0.0, 'FIT502': 0.0, 'FIT503': 0.0, 'FIT504': 0.0,
    'P501': 1.0, 'P502': 1.0, 'PIT501': 0.0, 'PIT502': 0.0, 'PIT503': 0.0,
    'FIT601': 0.0, 'P601': 1.0, 'P602': 2.0, 'P603': 1.0,
}

# Load model
print("Loading model...")
inference = ModelInference(model_path=r"D:\Fast Assignments\FYP\swat_fdai_model_final.pth")

# Test Normal Row
print("\n" + "=" * 60)
print("TEST 1: NORMAL ROW (label=0)")
print("=" * 60)
result = inference.predict(normal_row)
anomalies = result["anomalies"]
print(f"Anomalies detected: {len(anomalies)}/51 sensors")
if anomalies:
    for a in anomalies:
        print(f"  {a['node_id']:10s}  confidence={a['confidence']:.4f}  severity={a['severity']}")
else:
    print("  ✅ No anomalies detected — CORRECT for normal row")

# Stats from topology
topology = result["topology"]
probs = [n["confidence"] for n in topology["nodes"]]
mean_prob = sum(probs) / len(probs) if probs else 0
print(f"\nMean attack probability: {mean_prob:.4f}")
print(f"Min: {min(probs):.4f}, Max: {max(probs):.4f}")

# Test Attack Row
print("\n" + "=" * 60)
print("TEST 2: ATTACK ROW (label=1)")
print("=" * 60)
result = inference.predict(attack_row)
anomalies = result["anomalies"]
print(f"Anomalies detected: {len(anomalies)}/51 sensors")
if anomalies:
    for a in anomalies:
        print(f"  {a['node_id']:10s}  confidence={a['confidence']:.4f}  severity={a['severity']}")
else:
    print("  ❌ No anomalies detected — INCORRECT for attack row")

# Stats from topology
topology = result["topology"]
probs = [n["confidence"] for n in topology["nodes"]]
mean_prob = sum(probs) / len(probs) if probs else 0
print(f"\nMean attack probability: {mean_prob:.4f}")
print(f"Min: {min(probs):.4f}, Max: {max(probs):.4f}")

# Show process stages for attack nodes
print("\nAttacked sensors by process stage:")
for a in anomalies:
    node = next((n for n in topology["nodes"] if n["id"] == a["node_id"]), None)
    if node:
        print(f"  {a['node_id']:10s}  stage={node['stage']}")
