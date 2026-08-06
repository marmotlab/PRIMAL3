<div align="center">

<img src="assets/banner.png" alt="PRIMAL3: City-Scale Multi-Agent Path Finding with Up to 100,000 Agents." width="100%" />

<h1>PRIMAL3</h1>

<p><strong>Pathfinding via Reinforcement and Imitation Multi-Agent Learning</strong><br />
<em>Leveraging LaCAM3</em></p>

<p>Official PyTorch implementation and pretrained evaluation package for multi-agent pathfinding (MAPF).</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.8" />
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-2EA44F" alt="MIT License" /></a>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#pretrained-evaluation">Evaluation</a> ·
  <a href="#1000-agent-random-map-evaluation">1,000-agent run</a> ·
  <a href="#repository-layout">Repository layout</a>
</p>

</div>

## Overview

PRIMAL3 is a learning-based framework for multi-agent pathfinding (MAPF) that combines reinforcement learning with structured coordination. It targets difficult topological situations—such as bottlenecks, dead ends, and persistent conflicts—while retaining decentralized policy execution.

| Component | Purpose |
| --- | --- |
| **Topology-aware communication** | Represents compatible following relationships and competing paths using complementary interaction graphs. |
| **LaCAM3-guided training** | Uses confidence-triggered expert interventions and imitation targets for uncertain decisions. |
| **PIBT action refinement** | Applies persistent, learned, and distance-aware priorities to refine joint actions and prevent collisions. |
| **Cross-scale evaluation** | Includes input clipping and fast path extraction for evaluation beyond the training scale. |

This repository bundles the trained checkpoint, reproducible `32 × 32` benchmark instances for **50, 100, 150, 200, 250, and 300 agents**, and a separate `72 × 72` random-map set for **1,000-agent** evaluation.

## Quick start

### 1. Create the environment

The provided Conda environment uses Python 3.11 and includes the required PyTorch, Ray, scientific-computing, and visualization dependencies.

```bash
conda env create -f MAPF.yml
conda activate MAPF
```

> [!NOTE]
> Run all commands from the repository root. Evaluation paths are relative to this directory.

### 2. Run the default evaluation

```bash
python run_the_instances.py
```

The default configuration evaluates the pretrained model with 50 agents on 200 saved instances. Inference runs on CPU and test cases are parallelized with Ray.

## Pretrained evaluation

### Choose the number of agents

Set `EnvParameters.N_AGENTS` in [`alg_parameters.py`](alg_parameters.py):

```python
class EnvParameters:
    N_AGENTS = 150
```

The selected value must match one of the bundled instance sets:

| Agents | Instance file |
| ---: | --- |
| 50 | `32length_50agents_0.2density.pth` |
| 100 | `32length_100agents_0.2density.pth` |
| 150 | `32length_150agents_0.2density.pth` |
| 200 | `32length_200agents_0.2density.pth` |
| 250 | `32length_250agents_0.2density.pth` |
| 300 | `32length_300agents_0.2density.pth` |

All instance files are stored under `32_size_maps/32_32_0.2/`.

### Adjust evaluation resources

The main runtime settings are near the bottom of [`run_the_instances.py`](run_the_instances.py):

```python
ray.init(num_cpus=25)
num_runs = 200
```

- Lower `num_cpus` when fewer CPU cores are available.
- Lower `num_runs` for a shorter smoke test.
- Keep `num_runs` within the number of cases stored in the selected instance file.

After evaluation, the script reports:

- **success rate** — fraction of instances in which every agent reaches its goal;
- **average steps** — mean episode length across evaluated instances;
- **reach rate** — fraction of agents that reach their goals.

## 1,000-agent random-map evaluation

The [`random_1000` runner](32_size_maps/random_1000/run_the_instances.py) evaluates PRIMAL3 on 200 saved random-map instances. Each instance contains **1,000 agents** on a **`72 × 72` map** with obstacle density `0.2`.

### Configure the shared environment settings

The runner constructs the instance filename from `EnvParameters`, so first set these values in [`alg_parameters.py`](alg_parameters.py):

```python
class EnvParameters:
    N_AGENTS = 1000
    WORLD_SIZE = (72, 72)
    OBSTACLE_PROB = (0.19, 0.2)
```

This selects the bundled instance file:

```text
32_size_maps/random_1000/72length_1000agents_0.2density.pth
```

> [!NOTE]
> These values are shared with the standard evaluator. Restore `N_AGENTS = 50` and `WORLD_SIZE = (32, 32)` before returning to the default `32 × 32` evaluation.

### Run the evaluation

From the repository root, run all 200 cases:

```bash
python 32_size_maps/random_1000/run_the_instances.py
```

Use `--num_cases` for a shorter run:

```bash
python 32_size_maps/random_1000/run_the_instances.py --num_cases 5
```

The runner uses 25 Ray CPU workers by default and reports the same success-rate, average-step, and reach-rate metrics as the standard evaluator. Adjust `ray.init(num_cpus=25)` inside the runner if the machine has fewer available cores.

### Generate SVG animations

SVG generation is optional. Pass `--svg_save_dir` to save one animated trajectory for each evaluated case:

```bash
python 32_size_maps/random_1000/run_the_instances.py \
    --num_cases 5 \
    --svg_save_dir 32_size_maps/random_1000/svgs
```

The files follow this naming scheme:

```text
anim_<case-index>_<status>.svg
```

- `anim_0_success.svg` means every agent in case `0` reached its goal before the episode limit.
- `anim_12_fail.svg` means case `12` reached the episode limit before all agents arrived.

Each SVG is a self-contained, looping animation of one `72 × 72` episode:

| Element | Appearance |
| --- | --- |
| Obstacles | Dark grid cells |
| Agent goals | Hollow circles outlined with the corresponding agent color |
| Agents | Filled colored circles moving along their recorded trajectories |

Open an SVG directly in a modern web browser to play it. The existing [`svgs/`](32_size_maps/random_1000/svgs/) directory contains pre-generated animations for the saved cases.

> [!WARNING]
> A 1,000-agent SVG stores every agent position at every timestep. Individual files can be several megabytes, and the full 200-case directory can exceed 600 MB. Use a small `--num_cases` value when testing, or omit `--svg_save_dir` when only aggregate metrics are needed.

## Configuration reference

The central configuration lives in [`alg_parameters.py`](alg_parameters.py).

| Setting | Default | Description |
| --- | ---: | --- |
| `EnvParameters.N_AGENTS` | `50` | Number of agents used by the bundled evaluator. |
| `EnvParameters.WORLD_SIZE` | `(32, 32)` | Benchmark map dimensions. |
| `EnvParameters.OBSTACLE_PROB` | `(0.19, 0.2)` | Obstacle-density range; evaluation instances use `0.2`. |
| `EnvParameters.EPISODE_LEN` | `512` | Maximum number of steps per episode. |
| `EnvParameters.CLIP` | `True` | Clips scale-dependent observations to their training range. |
| `EnvParameters.FAST_PATHS` | `True` | Uses cached goal-BFS maps for large-scale path extraction. |
| `EnvParameters.PIBT_SVO_WEIGHT` | `1` | Weight of the learned social priority in PIBT shielding. |

The pretrained checkpoint used by the evaluator is located at:

```text
models/primal3/primal3_v22_pibt_inherit_v217-06-261404/26427392/net_checkpoint.pkl
```

## Repository layout

```text
.
├── alg_parameters.py       # Environment, network, and optimization settings
├── run_the_instances.py    # Parallel pretrained-model evaluation
├── sequence_test.py        # Hand-authored scenario runner and SVG export
├── mapf_gym.py             # MAPF environment and execution logic
├── model.py                # Model interface, inference, and optimization
├── net.py                  # Policy/value network
├── dual_comms.py           # Topology-aware agent communication
├── transformer.py          # Attention modules
├── pibt_shielding.py       # Priority-based action refinement
├── expert_guidance.py      # LaCAM3 expert integration
├── lacam3/                 # Bundled LaCAM3 Python bindings and source
├── pibt/                   # PIBT implementation
├── models/                 # Pretrained PRIMAL3 checkpoint
└── 32_size_maps/           # Saved benchmark instances
    └── random_1000/        # 72 × 72 / 1,000-agent runner, cases, and SVGs
```

## Troubleshooting

<details>
<summary><strong>Ray tries to start more workers than the machine can support</strong></summary>

Reduce `num_cpus` in `run_the_instances.py`. For a quick local check, also reduce `num_runs`.

</details>

<details>
<summary><strong>The selected instance file cannot be found</strong></summary>

Confirm that `N_AGENTS` is one of the six supported values and that the command is being run from the repository root.

</details>

<details>
<summary><strong>CUDA is unavailable during evaluation</strong></summary>

The pretrained evaluator explicitly loads its checkpoint onto CPU, so a GPU is not required for `run_the_instances.py`.

</details>

## License

This project is released under the [MIT License](LICENSE.txt).

## Contact

For questions about PRIMAL3, contact Chengyang He at [chengyanghe@u.nus.edu](mailto:chengyanghe@u.nus.edu) or [hecy@stanford.edu](mailto:hecy@stanford.edu).
