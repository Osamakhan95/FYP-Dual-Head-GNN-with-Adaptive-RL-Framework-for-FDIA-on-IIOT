# Dual-Head GNN with Adaptive Reinforcement Learning for FDIA Detection in Industrial IoT

## 📌 Overview

This project presents a **Dual-Head Graph Neural Network (GNN) with an Adaptive Reinforcement Learning (RL) framework** for detecting and responding to **False Data Injection Attacks (FDIAs)** in **Industrial Internet of Things (IIoT)** environments.

Industrial IoT systems rely on interconnected sensors, actuators, controllers, and communication networks. The injection of manipulated sensor data can compromise system reliability and lead to incorrect decisions. This project addresses this challenge by combining **graph-based deep learning** for attack detection with **reinforcement learning** for adaptive decision-making.

The proposed framework is designed to learn relationships between IIoT components and identify abnormal or malicious data while allowing the system to adapt its response based on the detected security state.

---

## 🎯 Research Problem

False Data Injection Attacks manipulate legitimate sensor measurements in an Industrial IoT environment. Since the injected values can appear statistically normal, traditional anomaly detection techniques may fail to identify them.

The objective of this research is to develop an intelligent security framework capable of:

- Modeling relationships between IIoT components.
- Detecting False Data Injection Attacks.
- Learning complex spatial and temporal patterns.
- Reducing false detection.
- Adapting security decisions using Reinforcement Learning.
- Providing an intelligent response to detected attacks.

---

## 🚀 Proposed Framework

The proposed framework combines two major components:

### 1. Dual-Head Graph Neural Network

The GNN component represents the IIoT environment as a graph consisting of nodes and relationships between them.

The Dual-Head architecture is designed to learn complementary representations for FDIA detection.

Depending on the experimental configuration, the architecture incorporates components such as:

- Graph Neural Networks / Graph Convolutional Networks
- Node feature embeddings
- Temporal feature extraction
- Feature fusion
- Attention mechanisms
- Dual prediction heads
- Binary attack classification

### 2. Adaptive Reinforcement Learning

The Reinforcement Learning component uses the detection state produced by the neural network to make adaptive security decisions.

The RL framework is designed around:

- State representation
- Actions
- Reward function
- Environment interaction
- Policy learning
- Adaptive decision-making

The overall objective is to improve the system's ability to respond to changing attack conditions.

---

## 🏗️ System Architecture

The high-level workflow is:

```text
             IIoT Sensor Data
                    │
                    ▼
            Data Preprocessing
                    │
                    ▼
          Graph Construction
                    │
                    ▼
        ┌───────────────────────┐
        │   Dual-Head GNN       │
        │                       │
        │  Graph Representation │
        │          +            │
        │  Feature Extraction   │
        └───────────┬───────────┘
                    │
                    ▼
             FDIA Detection
                    │
                    ▼
          Detection / Security
                 State
                    │
                    ▼
        ┌───────────────────────┐
        │ Adaptive Reinforcement│
        │      Learning          │
        └───────────┬───────────┘
                    │
                    ▼
          Adaptive Response
