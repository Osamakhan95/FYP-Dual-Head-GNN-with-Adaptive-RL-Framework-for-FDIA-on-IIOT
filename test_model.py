import pandas as pd
import torch
import sys
sys.path.insert(0, 'backend')

from app.utils.model_loader import SWATFDAIModel, ModelInference

print("=" * 70)
print("  TESTING SWAT FDAI MODEL WITH REAL DATA")
print("=" * 70)

# Load dataset
df = pd.read_csv("test SWaT_Dataset_Attack_v0.csv", header=1)

# Get sensor column names (strip whitespace)
raw_cols = df.columns.tolist()
sensor_cols = []
for col in raw_cols:
    stripped = col.strip()
    if stripped not in ["Timestamp", "Normal/Attack", ""]:
        sensor_cols.append(col)

print(f"Dataset: {len(df)} rows, {len(sensor_cols)} sensors")
print()

# Load model
model = ModelInference()
print()

# === TEST 1: Normal Row ===
print("TEST 1: NORMAL ROW (label=0, no attack)")
print("-" * 50)
normal_row = df[df.iloc[:, -1] == 0.0].iloc[0]
normal_data = {}
for col in sensor_cols:
    name = col.strip()
    val = float(normal_row[col])
    normal_data[name] = val

result = model.predict(normal_data)
anomalies = result["anomalies"]
topology = result["topology"]
print(f"  Anomalies detected: {len(anomalies)}")
if anomalies:
    for a in anomalies:
        print(f"    Node: {a['node_id']}, Confidence: {a['confidence']:.4f}, Severity: {a['severity']}")
else:
    print("  No anomalies detected (CORRECT - this is a normal row)")
print(f"  Topology nodes: {len(topology['nodes'])}")
print()

# === TEST 2: Attack Row ===
print("TEST 2: ATTACK ROW (label=1, FDI attack)")
print("-" * 50)
attack_row = df[df.iloc[:, -1] == 1.0].iloc[0]
attack_data = {}
for col in sensor_cols:
    name = col.strip()
    val = float(attack_row[col])
    attack_data[name] = val

result2 = model.predict(attack_data)
anomalies2 = result2["anomalies"]
topology2 = result2["topology"]
print(f"  Anomalies detected: {len(anomalies2)}")
if anomalies2:
    for a in anomalies2:
        print(f"    Node: {a['node_id']}, Confidence: {a['confidence']:.4f}, Severity: {a['severity']}")
else:
    print("  No anomalies detected")
print(f"  Topology nodes: {len(topology2['nodes'])}")
print()

# === Raw Predictions ===
print("RAW MODEL OUTPUT (per-node probabilities):")
print("-" * 50)

device = torch.device("cpu")
x_normal = torch.tensor([[v] for v in normal_data.values()], dtype=torch.float32)
x_attack = torch.tensor([[v] for v in attack_data.values()], dtype=torch.float32)

num_nodes = len(normal_data)
edges = []
for i in range(num_nodes - 1):
    edges.append([i, i + 1])
    edges.append([i + 1, i])
edges.append([0, num_nodes // 2])
edges.append([num_nodes // 2, 0])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

with torch.no_grad():
    pred_normal = model.model(x_normal, edge_index)
    pred_attack = model.model(x_attack, edge_index)

print("Normal row predictions (all 51 nodes):")
for i, (name, prob) in enumerate(zip(normal_data.keys(), pred_normal.flatten())):
    marker = " *** ATTACK" if prob > 0.5 else ""
    print(f"  {name}: {prob.item():.6f}{marker}")

print()
print("Attack row predictions (all 51 nodes):")
for i, (name, prob) in enumerate(zip(attack_data.keys(), pred_attack.flatten())):
    marker = " *** ATTACK" if prob > 0.5 else ""
    print(f"  {name}: {prob.item():.6f}{marker}")

print()
print("=" * 70)
print("SUMMARY:")
print("=" * 70)
normal_above = (pred_normal > 0.5).sum().item()
attack_above = (pred_attack > 0.5).sum().item()
print(f"  Normal row -> Mean: {pred_normal.mean().item():.6f}, Max: {pred_normal.max().item():.6f}, Nodes flagged: {normal_above}/{num_nodes}")
print(f"  Attack row -> Mean: {pred_attack.mean().item():.6f}, Max: {pred_attack.max().item():.6f}, Nodes flagged: {attack_above}/{num_nodes}")
print()
print("MODEL TEST COMPLETE - No errors!")
