# Adaptive Pathfinding Enemies in Tower Defense Games  
**Technical Scoping & Evaluation Plan**

**Team Members:** Xiaoqin Pi, Weiyuan Ding  

---

## Problem / Behavior

We aim to develop and evaluate adaptive enemy pathfinding systems for tower defense games that dynamically adjust routes based on tower placement and game state. Traditional TD games constrain enemies to fixed paths, limiting strategic depth.  

Our adaptive system will enable enemies to replan routes in real-time when towers are placed or upgraded, balance multiple objectives (minimizing distance vs. damage exposure), and coordinate movement in groups to avoid congestion or execute sacrificial strategies.  

This creates emergent gameplay where players must continuously adapt their defense strategies rather than memorizing optimal static placements.

---

## Game Engine or Development Platform

- **Development Environment:** Python 3.11+ with Pygame 2.0+  
- **Target Systems:** Cross-platform (macOS, Linux) with Python runtime  
- **Performance Requirements:**
  - **A\*** and **ACO:** 50+ enemies at 60 FPS  
  - **DQN:** 10–15 enemies at 30 FPS (or 50+ with decision caching/batching)

---

## AI Technique

We will implement and compare three pathfinding approaches:

### 1. Enhanced A* Search
Extends standard A* with composite cost functions incorporating geometric distance, expected damage exposure from towers, and local congestion metrics.  
→ Serves as our **interpretable baseline**.

### 2. Ant Colony Optimization (ACO)
Enables swarm intelligence where enemy groups deposit virtual pheromones on paths, influencing future routing decisions.  
→ Allows **emergent coordination** like route diversification and bottleneck avoidance.

### 3. Deep Q-Network (DQN)
A reinforcement learning approach where agents learn movement policies through experience.  
The state space encodes local map structure, tower positions, and damage zones.  
Actions correspond to movement directions.  
Rewards balance survival, path efficiency, and group success.

---

## Data Sources

We will generate and collect three types of data:

### 1. Game State Data (Runtime Input)
Real-time information fed to AI agents each frame.

| Frame | Enemy ID | Position | Tower Locations | Damage Zones | Path Cost |
|--------|-----------|-----------|-----------------|----------------|------------|
| 1250 | E023 | (5,12) | [(3,8),(7,10),(9,5)] | [(3,8,50),(7,10,75)] | 24.5 |
| 1250 | E024 | (6,13) | [(3,8),(7,10),(9,5)] | [(3,8,50),(7,10,75)] | 26.2 |

---

### 2. Pathfinding Decision Logs

| Timestamp | Algorithm | Enemy ID | Chosen Path | Alt Paths | Decision Time (ms) |
|------------|------------|-----------|--------------|------------|--------------------|
| 12.5s | A*-Enhanced | E023 | [A,F,K,P,Goal] | 3 | 0.8 |
| 12.5s | ACO | E024 | [A,B,G,L,Goal] | 5 | 1.2 |

---

### 3. Performance Metrics

| Wave | Algorithm | Enemies | Survived | Avg Damage | Path Diversity | CPU Time |
|------|------------|----------|-----------|--------------|----------------|-----------|
| 5 | Fixed | 30 | 12 (40%) | 180 | 1.0 | 0.1s |
| 5 | A*-Enhanced | 30 | 18 (60%) | 145 | 2.4 | 0.3s |
| 5 | ACO | 30 | 21 (70%) | 132 | 3.8 | 0.5s |

**Data Collection:** Automated logging during gameplay sessions. CSV export for offline analysis.

---

## Scale

- **A\*/ACO algorithms:** 50–100 simultaneous enemies per wave  
- **DQN algorithm:** 10–15 simultaneous enemies (or mixed deployment)  
- **Development:** ~1,000 enemy instances  
- **Final evaluation:** 10,000+ instances across 100+ matches  

---

## Adaptation Plan

### Baseline Implementation
- **Fixed-Path:** Enemies follow predefined shortest path, ignoring tower placement  
- **Basic A\*:** Standard A* with uniform edge costs, replanning only when path blocked  
- **Simple ACO:** Random pheromone initialization, basic deposit/evaporation rates  
- **Vanilla DQN:** Default hyperparameters, reward = +1 for survival, −1 for elimination  

### Planned Adaptations

#### Enhanced Cost Functions
- Damage risk assessment: `cost = distance + α·expected_damage + β·congestion`
- Dynamic weight tuning based on enemy health and wave progression  

#### Coordinated Pathfinding
- Group movement with formation constraints  
- Sacrificial unit logic: designate “tank” units to draw fire  
- Communication mechanisms for sharing discovered safe routes  

#### Learning Optimizations
- Curriculum learning: train DQN on progressively harder tower configurations  
- Experience replay prioritization based on surprise (high TD-error)  
- Ensemble methods: combine A* for short-term tactics with DQN for strategic routing  

#### Performance Enhancements
- Hierarchical pathfinding with sector-based planning  
- Path caching and incremental updates for dynamic environments  
- DQN optimization strategies:
  - Batch inference for groups of enemies (process 10 per batch)
  - Decision caching: reuse decisions for 5–10 frames
  - Hybrid approach: DQN for “leader” enemies, simpler following behavior for others
  - Model quantization to reduce inference time

> These adaptations transform basic algorithms into sophisticated systems capable of **emergent behavior** and **real-time adaptation**.

---

## Evaluation Plan

We will conduct comparative experiments across diverse scenarios.

### Test Maps
- Branching paths (2–3 route choices)  
- Open arena (maximum routing freedom)  
- Dynamic maze (walls appear/disappear)  

### Metrics
- **Survival Rate:** % of enemies reaching the goal  
- **Damage Efficiency:** Avg damage taken per surviving unit  
- **Path Diversity:** Shannon entropy of route distribution  
- **Adaptation Speed:** Frames to converge on new optimal path after tower placement  
- **Computational Cost:** Avg ms per pathfinding decision  
- **Strategic Depth:** # of distinct tower configurations that change enemy behavior  

---

### Success Criteria
- Adaptive algorithms achieve ≥25% higher survival rate than fixed paths  
- Path diversity index ≥ 2.0 (indicating meaningful route variation)  
- Real-time performance maintained:
  - A*/ACO: <5ms per enemy at 60 FPS (50+ enemies)
  - DQN: <33ms per decision (10–15 active DQN agents)

### Failure Criteria
- Adaptive algorithms perform worse than fixed paths in >50% of scenarios  
- Pathfinding causes frame drops below 30 FPS with 50+ enemies  
- Emergent behaviors are chaotic/unpredictable rather than strategic  
- Implementation complexity prevents completion within project timeline  

---

## Layered Development Schedule

### Functional Minimum
- Fork and extend Pygame TD template with configurable maps *(Xiaoqin)*  
- Implement basic A* pathfinding with grid-based navigation *(Weiyuan)*  
- Create data logging infrastructure for game state and decisions *(Xiaoqin)*  
- Fixed-path baseline enemy behavior for comparison *(Weiyuan)*  

### Low Target
- Enhanced A* with composite cost function *(Weiyuan)*  
- Real-time replanning system triggered by tower placement *(Xiaoqin)*  
- Basic ACO implementation with pheromone deposit/evaporation *(Weiyuan)*  
- Performance profiling and optimization for 50+ enemies *(Xiaoqin)*  
- Initial evaluation on linear and branching path maps *(Both)*  

### Desired Target
- **Advanced A\* Features:**
  - Congestion avoidance using density maps *(Weiyuan)*  
  - Formation-based group movement *(Xiaoqin)*  
- **Refined ACO:** Parameter tuning for emergent behaviors *(Xiaoqin)*  
- **DQN Implementation with Hybrid Deployment:**
  - Offline training pipeline using *stable-baselines3* *(Weiyuan)*  
  - Leader-follower architecture (5–10 DQN leaders, 40+ A* followers) *(Weiyuan)*  
  - Decision batching and caching system *(Xiaoqin)*  
- Comprehensive evaluation suite with automated testing *(Xiaoqin)*  
- Real-time visualization overlays for debugging and demos *(Weiyuan)*  

### High Target
- **Advanced Coordination Mechanisms:**
  - Sacrificial unit strategies with role assignment *(Xiaoqin)*  
  - Multi-objective Pareto optimization for path selection *(Weiyuan)*  
- Hierarchical pathfinding with sector-level planning *(Weiyuan)*  
- DQN transfer learning experiments across map types *(Xiaoqin)*  
- Statistical significance testing and performance analysis *(Weiyuan)*  

### Extras
- Adversarial tower AI using genetic algorithms *(Xiaoqin)*  
- WebGL port for browser-based gameplay *(Weiyuan)*  
- Open-source release with documentation *(Both)*  
