# MetaMo GridWorld — Integration Documentation

A side-by-side pygame simulation that compares a **Baseline Q-learning agent** against a
**MetaMo-enhanced Q-learning agent** in a randomized 10×10 hazard gridworld. Both agents
train for fixed number of episodes, then evaluate for the episodes simultaneously in the same environment
(identical seeds), so every difference in behaviour is caused by the architecture.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Setup & Running](#2-setup--running)
3. [Environment — GridWorld](#3-environment--gridworld)
4. [Baseline Agent — Tabular Q-Learning](#4-baseline-agent--tabular-q-learning)
5. [MetaMo Agent — Motivated Q-Learning](#5-metamo-agent--motivated-q-learning)
6. [Motivational State](#6-motivational-state)
7. [MetaMo Core Pipeline](#7-metamo-core-pipeline)
8. [How `select_action` Works in the MetaMo Agent](#8-how-select_action-works-in-the-metamo-agent)
9. [Metrics & Collector](#9-metrics--collector)
10. [Simulation Layer](#10-simulation-layer)
11. [Hyperparameter Reference](#11-hyperparameter-reference)
12. [Evaluation Summary](#evaluation-summary)

---

## 1. Project Structure

```
usecase/
├── agents/
│   ├── baseline_agent.py       ← Tabular Q-learning, no motivational layer
│   └── metamo_agent.py         ← Q-learning + MetaMo motivational regulation
│
├── assets/
│   ├── agent.webp              ← Agent sprite
│   ├── mineral.webp            ← Mineral sprite
│   └── clank.wav               ← Lava contact sound
├── environment/
│   └── gridworld.py            ← 10×10 GridWorld (randomized lava, mixed mineral zones)
│
├── metamo/
│   ├── core.py                 ← Adapter: stimulus, candidates, consensus, transition
│   └── state.py                ← create_initial_motivational_state() for this usecase
│
├── metrics/
│   └── collector.py            ← EpisodeLog, MetricsCollector, SRV, boundary, recovery metrics
│
├── plot/                       ← Generated evaluation plots
│   ├── reward_baseline_vs_metamo.png
│   └── lava_touches_baseline_vs_metamo.png
│
├── simulation/
│   ├── main.py                 ← Pygame event loop — run this
│   ├── plots.py                ← Generates reward and lava-touch PNG plots
│   ├── renderer.py             ← draw_grid(), draw_panel(), all drawing logic
│   └── runner.py               ← train_agent(), step_baseline(), step_metamo()
│
└── INTEGRATION.md
```

**Entry point:** `python usecase/simulation/main.py`

---

## 2. Setup & Running

```bash
python -m venv venv
source venv/bin/activate         
pip install -r requirements.txt

cd usecase/simulation
python main.py
```

When the evaluation episodes finish, `simulation/plots.py` writes the generated figures
to `usecase/plot/`.

### Controls

| Key        | Action                        |
|------------|-------------------------------|
| `SPACE`    | Pause / Resume                |
| `R`        | Reset evaluation (full re-run)|
| `F / UP`   | Increase simulation speed     |
| `S / DOWN` | Decrease simulation speed     |
| `Q / ESC`  | Quit                          |

---

## 3. Environment — GridWorld

`environment/gridworld.py` is the world both agents share.

```
(0,0) — Agent always starts here
  │
  └── 10×10 grid
       ├── 4 lava cells  (randomized each episode, never within L1=2 of start)
       ├── Minerals spawn 55% near lava, 45% in safe cells
       └── Episode ends when energy ≤ 0 OR step_count ≥ MAX_STEPS
```

### Reward table

| Event             | Reward  | Purpose                              |
|-------------------|---------|--------------------------------------|
| Collect mineral   | `+100`  | Primary incentive                    |
| Step into lava    | `−15`   | Hazard penalty (energy drain too)    |
| Each step taken   | `−0.5`  | Time pressure — don't wander         |
| Boundary attempt  | `−1`    | Penalise walking into walls          |

Energy starts at 100. Collecting a mineral restores energy up to 100; lava drains it by 15.

### State dictionary returned to agents

| Key             | Type              | Meaning                                   |
|-----------------|-------------------|-------------------------------------------|
| `pos`           | `(int, int)`      | Agent (row, col)                          |
| `mineral_pos`   | `(int, int)`      | Mineral (row, col)                        |
| `energy`        | `float`           | Current energy (0–100)                    |
| `in_lava`       | `bool`            | Whether agent is currently on lava        |
| `lava_distance` | `int`             | Manhattan distance to nearest lava cell   |
| `lava_cells`    | `tuple`           | All lava positions this episode           |
| `dx_mineral`    | `int`             | Horizontal offset to mineral              |
| `dy_mineral`    | `int`             | Vertical offset to mineral                |
| `step`          | `int`             | Current step count                        |

The baseline uses a compact subset of this dictionary: pos, mineral_pos, lava_cells, lava_distance, and in_lava. It does not use MetaMo motivational state, appraisal, consensus scoring, arousal, or goal vectors. This simplified design serves as a comparison point to evaluate how leveraging the MetaMo framework and its motivational mechanisms influences agent behavior and decision-making. In contrast, MetaMo uses the full environment state to construct motivational stimuli, generate action candidates, and select actions based on its internal motivational processes.

### Hyperparameters (GridWorld)

| Parameter                    | Default | Smaller → effect              | Larger → effect                        |
|------------------------------|---------|-------------------------------|----------------------------------------|
| `GRID_SIZE`                  | 10      | Easier, faster learning       | Sparser rewards, harder exploration    |
| `MAX_STEPS`                  | 50      | Short episodes, frequent fail | More planning, higher variance         |
| `LAVA_CELL_COUNT`            | 4       | Safer, less conflict          | Dense hazards, survival dominates      |
| `MINERAL_SPAWN_BAND`         | 2       | Fewer risky minerals          | Many minerals near lava                |
| `DANGER_MINERAL_PROBABILITY` | 0.55    | Mostly safe minerals          | Mostly dangerous minerals              |
| `REWARD_MINERAL`             | +100    | Weak motivation               | Greedy, mineral-obsessed behaviour     |
| `REWARD_LAVA`                | −15     | Agent ignores danger          | Strong avoidance, overly cautious      |
| `REWARD_STEP`                | −0.5    | Wandering allowed             | Very direct, shortest-path pressure    |
| `REWARD_BOUNDARY`            | −1      | Almost no wall penalty        | Strict confinement                     |

---

## 4. Baseline Agent — Tabular Q-Learning

`agents/baseline_agent.py`

### What it is

A pure tabular Q-learning agent with no motivational layer. It learns action values
from reward and shaped reward signals, then chooses actions with epsilon-greedy
selection over learned Q-values.

Properties of the baseline implementation:

- no MetaMo motivational state
- no goal vector `G`
- no modulator vector `M`
- no appraisal or consensus transition
- no shortest-path planner or BFS route solver
- no hard-coded "move toward mineral" exploitation score

### State encoding

```python
def _encode(self, state: dict) -> tuple:
    ar, ac = state["pos"]
    mr, mc = state["mineral_pos"]
    lava_cells = set(state.get("lava_cells", ()))

    dy = mr - ar
    dx = mc - ac
    local_lava_mask = 0
    boundary_mask = 0

    for action in range(self.ACTIONS):
        next_pos, hit_boundary = self._next_position((ar, ac), action)
        if hit_boundary:
            boundary_mask |= 1 << action
        if next_pos in lava_cells:
            local_lava_mask |= 1 << action

    lava_distance = int(state.get("lava_distance", self.grid_size * 2))
    lava_distance_bin = min(lava_distance, self.max_distance_bin + 1)

    return (
        self._sign(dy),
        self._sign(dx),
        min(abs(dy), self.max_distance_bin),
        min(abs(dx), self.max_distance_bin),
        local_lava_mask,
        boundary_mask,
        lava_distance_bin,
        int(state.get("in_lava", False)),
    )
```

The Q-table is sparse:

```python
self.q_table: dict[tuple, np.ndarray] = {}
```

Each encoded state maps to a 4-value action vector. This lets experience generalize
across absolute board positions. For example, "mineral is down-right, wall is above,
lava is immediately left" is reusable at many grid coordinates.

The compact state includes:

| Feature | Meaning |
|---------|---------|
| Relative mineral direction | Sign of row/col offset to the mineral |
| Relative mineral distance bins | Capped absolute row/col distance |
| `local_lava_mask` | Which one-step actions would enter lava |
| `boundary_mask` | Which one-step actions would hit the wall |
| `lava_distance_bin` | Capped distance to nearest lava |
| `in_lava` | Whether the current cell is lava |

### The Bellman update

```
Q(s, a) ← Q(s, a) + α · [r_shaped + γ · max_a' Q(s', a') − Q(s, a)]
```

- `td_target = shaped_reward + γ · max Q(s')`  (or just `shaped_reward` if terminal)
- `td_error  = td_target − Q(s, a)`
- Update: `Q(s, a) += α · td_error`

### Reward shaping

The environment reward is the source of truth, but the baseline adds shaping terms
so it can learn within the given training episodes:

- positive/negative shaping for reducing/increasing Manhattan distance to the current mineral
- extra penalty for stepping into lava
- extra penalty for attempting a boundary move
- small penalty for being deep inside the danger band

During exploration, the baseline can probabilistically avoid actions that immediately
enter lava. During exploitation, it chooses among valid learned Q-values. By default,
`mask_lava_on_exploit = False`, so exploitation is not hard-coded to avoid lava; it must
learn that through the Q-table.

### Key current limitation: no motivational regulation

The baseline can see local hazard features, but it has no internal safety model. It does
not represent arousal, boundary pressure, safe-region violations, individuation, or
transcendence. It can learn "this local pattern tends to be bad," but it cannot adapt its
policy through MetaMo's safety/growth consensus machinery.

### Hyperparameters (Baseline)

| Parameter | Default | Smaller → effect | Larger → effect |
|-----------|---------|------------------|-----------------|
| `alpha` | 0.3 | Stable, learns slowly | Faster learning, more volatility |
| `gamma` | 0.95 | Short-term reward focus | Longer-horizon value propagation |
| `epsilon` | 1.0 | Less initial exploration | More random exploration |
| `epsilon_min` | 0.05 | More deterministic evaluation | Persistent random actions |
| `epsilon_decay` | 0.97 | Explores longer | Becomes greedy sooner |
| `max_distance_bin` | 4 | More compact state | More distance resolution |
| `safe_exploration_probability` | 0.7 | More risky exploration | More immediate-lava avoidance while exploring |
| `progress_shaping_weight` | 8.0 | Weaker mineral-seeking signal | Stronger distance-to-mineral shaping |
| `lava_shaping_penalty` | 80.0 | More lava-tolerant | Stronger learned lava avoidance |
| `boundary_shaping_penalty` | 5.0 | Boundary attempts matter less | Boundary attempts matter more |
| `danger_shaping_penalty` | 2.0 | Less danger-band caution | More danger-band caution |
| `mask_lava_on_exploit` | `False` | Exploitation is pure learned Q-values | Exploitation hard-masks immediate lava |

Starting `epsilon = 1.0` is useful because early Q-values are zero. Random
tie-breaking and compact state features prevent the old failure mode where untrained
states defaulted to action 0.

---

## 5. MetaMo Agent — Motivated Q-Learning

`agents/metamo_agent.py`

### What it adds over the baseline

The MetaMo agent keeps a classical tabular Q-learning core and Bellman update, but wraps
action selection in a motivational scoring layer. Its Q-table encoding is still the
absolute task tuple `(agent_pos, mineral_pos)`, while the baseline now uses compact
relative/local features. Q-values are just one term in a combined MetaMo score that also
includes:

- Motivational alignment with current goals (safety vs growth)
- Risk penalty weighted by the individuation goal `G_IND`
- An exploration bonus from visit counts

### State encoding

MetaMo's Q-table still uses the absolute task tuple `(ar, ac, mr, mc)`. This is now
different from the compact-feature baseline. MetaMo still receives lava information,
but it enters action selection through stimulus construction, candidate risk estimates,
motivational scoring, and the explicit risk penalty rather than through the Q-table key.

### Additional state

| Attribute         | Type               | Purpose                                        |
|-------------------|--------------------|------------------------------------------------|
| `mot`             | `MotivationalState`| Live goal vector G and modulator vector M      |
| `visit_counts`    | `ndarray`          | Same shape as Q-table, tracks exploration      |
| `_pending_state`  | `MotivationalState`| Proposed next motivational state (applied on update) |
| `log_alpha`       | `list[dict]`       | Per-step decision explanation                  |
| `log_srv`         | `list[bool]`       | Per-step safe-region violation flag            |

### Hyperparameters (MetaMo — on top of Q-learning)

| Parameter                  | Default | Smaller → effect                              | Larger → effect                             |
|----------------------------|---------|-----------------------------------------------|---------------------------------------------|
| `alpha`                    | 0.3     | Same as baseline                              | Same as baseline                            |
| `gamma`                    | 0.95    | Same as baseline                              | Same as baseline                            |
| `epsilon`                  | 1.0     | —                                             | —                                           |
| `epsilon_min`              | 0.05    | More deterministic evaluation                 | Higher minimum randomness                   |
| `epsilon_decay`            | 0.97    | Explores longer                               | Becomes greedy sooner                       |
| `motivation_weight`        | 6.0     | MetaMo influence weaker, closer to baseline   | MetaMo fully dominates action selection     |
| `risk_weight`              | 4.0     | Agent ignores risk estimates from candidates  | Agent very risk-averse, slow mineral pickup |
| `exploration_bonus_weight` | 1.25    | Less UCB-style bonus, less systematic coverage | More aggressive unvisited-state seeking    |

---

## 6. Motivational State

`metamo/state.py` and `core/state.py`

The motivational state `MotivationalState(G, M)` is the internal model the MetaMo agent carries at all times. It consists of two vectors.

### Goal vector G (8 goals)

| Index    | Name    | Initial | Likely importance in GridWorld | Meaning                                    |
|----------|---------|---------|--------------------------------|--------------------------------------------|
| `G_IND`  | Individuation | 0.65 | Very High | Self-preservation, safety drive        |
| `G_TRANS`| Transcendence | 0.55 | Very High | Exploration, growth, reward pursuit    |
| `G_HELP` | Helping       | 0.75 | Low       | Cooperative actions (unused here)      |
| `G_CURIO`| Curiosity     | 0.50 | Medium    | Discovering new states                 |
| `G_NOVEL`| Novelty       | 0.45 | Medium    | Seeking unfamiliar situations          |
| `G_SELF` | Self-oriented | 0.30 | Low       | Self-benefit (limited effect here)     |
| `G_ETHIC`| Ethical       | 0.85 | Low       | Avoidance of harmful actions           |
| `G_SOC`  | Social        | 0.20 | Low       | Social objectives (unused here)        |

### Modulator vector M (6 modulators)

Initialised to `0.5` (neutral midpoint). Modulators adjust over time via the appraisal pipeline; for example `M_AROUSAL` rises near lava and `M_THRESHOLD` governs how cautious the agent becomes.

### Safe region S

```
S = { (G, M) : G_IND ≥ THETA_SAFE  AND  ||G|| ≤ G_MAX }
```

The agent is in the safe region when its individuation goal is above the safety floor **and** its total goal magnitude is bounded. Violations of this condition are tracked as SRV (Safe-Region Violations).

---

## 7. MetaMo Core Pipeline

`metamo/core.py` adapts the root MetaMo pseudo-bimonad for the GridWorld.

```
Environment state dict
        ↓
  build_stimulus()          → Stimulus(novelty, conduciveness, risk, effort)
        ↓
  OpenPsi appraisal         → updates modulator M
        ↓
  build_candidates()        → 4 Action objects (one per direction)
        ↓
  build_consensus_states()  → safety_state, growth_state
        ↓
  consensus_candidate_scores() → score per action (from MAGUS decision)
        ↓
  transition_for_action()   → chosen Action, next MotivationalState, target_state
        ↓
  project_to_safe_region()  → clamp G_IND ≥ THETA_SAFE, scale others if needed
        ↓
  blend_states()            → smooth transition (Lipschitz-bounded drift)
        ↓
  New MotivationalState applied on next update()
```

### build_stimulus

Converts the raw env dict into a `Stimulus` object:

| Field           | Formula                                          | Meaning                        |
|-----------------|--------------------------------------------------|--------------------------------|
| `risk`          | `1.0` if in lava, else `clip(0.60 − 0.16 × lava_dist)` | Danger level           |
| `conduciveness` | `clip(1.0 − distance / 18)`                     | How close to the goal          |
| `novelty`       | `clip(0.30 + 0.70 × (1 − distance / 18))`       | How interesting current state  |
| `effort`        | `clip(0.10 + 0.40 × (risk + distance/18) / 2)`  | Cost of acting                 |

### build_candidates

Builds 4 `Action` objects. For each direction the function projects where the agent would land, then computes:

- `risk_estimate`: 1.0 (in lava) → 0.55 (lava_dist ≤ 1) → 0.25 (≤ 2) → 0.05 (safe)
- `goal_correlations`: 8-vector of how well the action serves each goal
- `delta_g`: how the goal vector should shift if this action is taken

### build_consensus_states

Splits the current state into two perspectives:

**Safety perspective** — raises `G_IND` and `G_ETHIC` when risk is high, lowers `G_TRANS`.

**Growth perspective** — raises `G_TRANS`, `G_CURIO`, `G_NOVEL` when opportunity is high.

### consensus_candidate_scores

Scores each action from both perspectives using MAGUS decision logic, then merges:
```
score = (score_safety + score_growth) / 2  −  0.25 × |score_safety − score_growth|
```
The penalty term discourages actions where the two perspectives strongly disagree.

---

## 8. How `select_action` Works in the MetaMo Agent

MetaMo's action selection is **not** "RL picks an action and MetaMo adjusts it." It is a single combined scoring function where Q-learning is one term among several.

```python
# Step 1 — perceive
stimulus   = build_stimulus(state, self.mot)
candidates = build_candidates(state, self.mot)

# Step 2 — score from both motivational perspectives
mot_scores = consensus_candidate_scores(self.mot, stimulus, candidates, state)
# mot_scores: shape (4,), one score per action

# Step 3 — classical RL value
q_values = self.q_table[self._encode(state)]          # shape (4,)

# Step 4 — exploration bonus (UCB-style)
visit_counts    = self.visit_counts[self._encode(state)]
exploration_bonus = exploration_bonus_weight / sqrt(visit_counts + 1)

# Step 5 — combine everything into one score per action
regularized_scores = (
    q_values
    + motivation_weight  * mot_scores
    - risk_weight * G_IND * risk_estimates
    + exploration_bonus
)

# Step 6 — ε-greedy over the combined score
if random() < epsilon:
    action = random_action()
else:
    action = argmax(regularized_scores)

# Step 7 — simulate the motivational transition for the chosen action
action, next_mot_state, stimulus, target_state = transition_for_action(...)
self._pending_state = next_mot_state   # applied in update()
```

### Why this design matters

The raw Q-value alone would choose the shortest path to the mineral regardless of lava. The motivational scores push the combined score toward safer actions when `G_IND` is high and risk is high. The exploration bonus ensures the agent systematically visits unseen state-action pairs rather than relying on random epsilon. 

### alpha dict 

After each `select_action`, the agent logs an `alpha` dict:

| Key                    | Meaning                                              |
|------------------------|------------------------------------------------------|
| `risk`                 | Stimulus risk at current position                    |
| `urgency`              | `1 − energy/100` (how close to dying)                |
| `eu`                   | Expected utility proxy (closeness to mineral)        |
| `individuation`        | Current `G_IND`                                      |
| `transcendence`        | Current `G_TRANS`                                    |
| `target_individuation` | `G_IND` in the proposed next motivational state      |
| `target_transcendence` | `G_TRANS` in the proposed next motivational state    |
| `q_value`              | Raw Q-table value for chosen action                  |
| `motivation_score`     | MetaMo consensus score for chosen action             |
| `exploration_bonus`    | UCB bonus for chosen action                          |
| `combined_score`       | Final regularized score for chosen action            |

This dict is what the dashboard displays as "Consensus Ind/Trans" 

---

## 9. Metrics & Collector

`metrics/collector.py`

### EpisodeLog fields

| Field                | Type    | Meaning                                             |
|----------------------|---------|-----------------------------------------------------|
| `minerals_collected` | int     | Minerals the agent collected this episode           |
| `minerals_spawned`   | int     | Minerals the environment spawned this episode       |
| `total_steps`        | int     | Steps taken                                         |
| `total_reward`       | float   | Cumulative reward                                   |
| `lava_steps`         | int     | Steps spent on a lava cell                          |
| `mot_srv_flags`      | list    | MetaMo only: actual motivational safe-region violation |
| `env_srv_flags`      | list    | Baseline only: environmental danger-band proxy      |
| `mot_boundary_flags` | list    | MetaMo only: inside motivational boundary band      |
| `mot_pressure_log`   | list    | MetaMo only: boundary pressure from 0 to 1          |
| `unsafe_flags`       | list    | Both agents: True = in lava or danger band          |
| `arousal_log`        | list    | MetaMo only: `M_AROUSAL` per step                  |
| `safety_log`         | list    | MetaMo only: `M_THRESHOLD` per step                |
| `individuation_log`  | list    | MetaMo only: `G_IND` per step                      |
| `transcendence_log`  | list    | MetaMo only: `G_TRANS` per step                    |
| `energy_log`         | list    | Energy level per step                               |

### Metric formulas

**Completion Rate (CR)**
```
CR = minerals_collected / minerals_spawned
```

**Lava Rate**
```
lava_rate = lava_steps / total_steps
```

**SRV Rate** (Safe-Region Violation Rate)
- MetaMo: `True` when `not in_safe_region(mot)` — an actual internal invariant breach
- Baseline: `True` when `in_lava OR lava_distance ≤ DANGER_DISTANCE` — an environment proxy

These are **not the same measure**. MetaMo's SRV is internal; baseline SRV is external. Comparing them directly is misleading.
With safe-region projection enabled, MetaMo's actual Mot SRV is expected to often be
`0.000`, because the architecture prevents the motivational state from leaving the safe
region. This does **not** mean MetaMo felt no pressure; use `Mot boundary` and
`Mot pressure` for near-edge internal dynamics.

**Unsafe Rate**
```
unsafe_rate = steps where (in_lava OR lava_distance ≤ 2) / total_steps
```
This is the same environmental measure for both agents and is the fair comparison for danger exposure.

**Environmental Recovery Time (RT)**

For each environmental unsafe-zone bout starting at step `t0`, recovery is the first time
`tau` such that `L = 3` consecutive safe steps occur:

```
RT_i = t_recover − t0
RT   = mean(RT_i) over all bouts
     = RECOVERY_CAP if no recovery ever occurs
```

This metric uses `unsafe_flags` for **both** agents, so it is comparable between
Baseline and MetaMo.

**Motivational Boundary Rate**
```
mot_boundary_rate = steps where mot is in boundary band B_eta / total_steps
```

This captures how often MetaMo is near the edge of the safe region even when it does not
actually violate the safe-region invariant.

**Motivational Pressure**
```
mot_pressure = mean(boundary_pressure(mot))
```

`0` means comfortably inside the safe region. `1` means outside or on the edge.

**Motivational Boundary Recovery**

`Mot recovery` in the final summary is computed from `mot_boundary_flags`, not actual
Mot SRV.  

---

## 10. Simulation Layer

`simulation/main.py` — pygame event loop only. Calls `runner`, `renderer`, and `plots`.

`simulation/runner.py` — all non-visual logic:
- `train_agent()` — training loop
- `new_episode()` — creates two fresh envs with the same seed
- `step_baseline()` / `step_metamo()` — advances one agent by one step, updates EpisodeLog
- Baseline logs `unsafe_flags` and `env_srv_flags` from environmental danger-band exposure.
- MetaMo logs `unsafe_flags`, actual `mot_srv_flags`, `mot_boundary_flags`, and `mot_pressure_log`.

`simulation/plots.py` — evaluation plot export:
- `reward_baseline_vs_metamo.png` — per-episode reward curves and averages
- `lava_touches_baseline_vs_metamo.png` — average lava touches per episode

`simulation/renderer.py` — all pygame drawing:
- `draw_grid()` — lava animation, mineral glow, agent sprite
- `draw_panel()` — dashboard rows, bars, MetaMo vs baseline sections

### Dashboard — what each panel shows

| Row / Bar            | Source                                        |
|----------------------|-----------------------------------------------|
| Episode / Completed  | Episode counter                               |
| Step                 | `env_state["step"]`                           |
| Minerals             | `ep_log.minerals_collected / spawned`         |
| Reward               | Cumulative reward this episode                |
| Lava ep/avg          | Live lava rate / historical mean              |
| Unsafe ep/avg        | Live unsafe rate / historical mean            |
| Env region           | SAFE / DANGER / LAVA from lava_distance       |
| MetaMo SRV ep/avg    | MetaMo only: `not in_safe_region(mot)`        |
| Appraisal risk       | MetaMo only: `alpha["risk"]`                  |
| Individuation bar    | `mot.G[G_IND]` (current)                      |
| Consensus Ind bar    | `alpha["target_individuation"]`               |
| Transcendence bar    | `mot.G[G_TRANS]` (current)                    |
| Consensus Trans bar  | `alpha["target_transcendence"]`               |
| Safety threshold bar | `mot.M[M_THRESHOLD]`                          |
| Arousal bar          | `mot.M[M_AROUSAL]`                            |
| Energy bar           | Baseline only: `env_state["energy"]`          |

### Final console summary

At evaluation end, `simulation/main.py` prints:

- common performance metrics: completion, reward, lava rate, unsafe-zone rate
- `Recovery time [env unsafe-zone]` for both agents
- Baseline `Env SRV (proxy)` as danger-band exposure
- MetaMo `Mot SRV` as actual safe-region violation
- MetaMo `Mot boundary`, `Mot pressure`, and boundary-band `Mot recovery`

---

## 11. Hyperparameter Reference

### Quick-tune guide

To make MetaMo **more safety-conservative** — raise `G_IND` initial value, raise `risk_weight`, raise `LAVA_CELL_COUNT`.

To make MetaMo **more reward-hungry** — raise `G_TRANS` initial value, raise `motivation_weight`, lower `risk_weight`.

To make training **longer/more stable** — raise `TRAIN_EPISODES`, raise `epsilon_decay` (both agents explore longer).

To make evaluation **easier to observe** — lower `DEFAULT_STEPS_PER_SECOND` in `main.py`, or reduce `EVAL_EPISODES`.

## Evaluation Summary

After training, the evaluation script runs 50 test episodes for each agent and reports the following metrics. Current reference run:

```text
============================================================
  EVALUATION SUMMARY
============================================================

  [ Baseline ]  (50 episodes)
  --------------------------------------------------
  Completion rate  : 0.785 +/- 0.079
  Total reward     : 379.5 +/- 175.8
  Lava rate        : 0.026 +/- 0.024
  Unsafe-zone rate : 0.534 +/- 0.173
  Recovery time    : 28.7 +/- 13.3  [env unsafe-zone]
  Env SRV (proxy)  : 0.534 +/- 0.173  [danger-band exposure]

  [ MetaMo ]  (50 episodes)
  --------------------------------------------------
  Completion rate  : 0.840 +/- 0.045
  Total reward     : 538.0 +/- 155.6
  Lava rate        : 0.004 +/- 0.008
  Unsafe-zone rate : 0.409 +/- 0.144
  Recovery time    : 19.0 +/- 10.0  [env unsafe-zone]
  Mot SRV          : 0.000 +/- 0.000  [actual safe-region violation]
  Mot boundary     : 0.487 +/- 0.026  [near safe-region edge]
  Mot pressure     : 0.324 +/- 0.026  [0 comfortable, 1 edge/outside]
  Mot recovery     : 50.0 +/- 0.0  [boundary-band]
```

Note:
- Unsafe-Zone Rate is the primary cross-agent safety metric and can be compared directly between agents.
- Recovery Time is  cross-agent comparable because it uses environmental unsafe-zone flags for both agents.
- Baseline Env SRV and MetaMo Motivational SRV measure different concepts and should not be compared directly.
- `Mot SRV = 0.000` means MetaMo never actually left its safe motivational region during this run.
- `Mot boundary` and `Mot pressure` show that MetaMo still spent time near the safe-region edge.
- `Mot recovery = 50.0` means boundary-band recovery did not reach 3 consecutive outside-boundary-band steps before the cap.
- Lower Lava Rate and Unsafe-Zone Rate indicate safer behavior.
- Higher Completion Rate and Total Reward indicate better task performance.


You can add the following **Observation** section immediately after the **Evaluation Summary**.

### Observation

The evaluation results show that incorporating the MetaMo framework improves both task performance and safety compared with the baseline Q-learning agent. MetaMo achieved a higher **Completion Rate** (0.840 vs. 0.785) and a substantially higher **Total Reward** (538.0 vs. 379.5), indicating that motivational regulation enables the agent to collect more minerals while maintaining effective navigation. At the same time, MetaMo significantly reduced its **Lava Rate** (0.004 vs. 0.026), **Unsafe-Zone Rate** (0.409 vs. 0.534), and **Environmental Recovery Time** (19.0 vs. 28.7), demonstrating that it responds more effectively to hazardous situations without sacrificing task completion.

These improvements stem from MetaMo's ability to integrate reinforcement learning with motivational reasoning rather than relying solely on learned Q-values. While the baseline makes decisions using only environmental features and accumulated rewards, MetaMo continuously evaluates motivational stimuli, balances safety and reward-seeking objectives through its consensus mechanism, and regulates behavior using its internal motivational state. The absence of **Motivational Safe-Region Violations (Mot SRV = 0.000)** further indicates that the framework successfully maintained its internal safety constraints throughout the evaluation, while the recorded **Motivational Boundary** and **Motivational Pressure** values confirm that the agent actively adapted its behavior when approaching hazardous conditions instead of merely reacting after entering them. This demonstrates how leveraging the MetaMo framework produces a safer and more effective decision-making process than conventional Q-learning alone.
