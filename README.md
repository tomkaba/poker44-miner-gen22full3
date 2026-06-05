# gen22full3

Minimal release repository for Poker44 miner runtime scoring.

This repository is a standalone miner variant prepared for production rollout with the gen22_hybrid_full_0.0595 checkpoint, vote101 runtime aggregation, a minichunk threshold of 0.0595, and a parent cutoff of 66/101.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-gen22full3.git
cd poker44-miner-gen22full3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Miner

```bash
python neurons/miner.py
```

or legacy wrapper:

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

## Implementation

- Launcher: start_miner.sh
- Scorer entrypoint: poker44/miner_heuristics.py
- Entry point: neurons/miner.py
- Runtime model: weights/gen22_hybrid_full_0.0595.pt

Base release lineage: gen20tens1 with the runtime replaced by gen22_hybrid_full_0.0595 using deterministic vote101 aggregation and a 66/101 cutoff.

Manifest implementation SHA256 is computed from:

- start_miner.sh
- weights/gen22_hybrid_full_0.0595.pt
- neurons/miner.py
- poker44/__init__.py
- poker44/base/miner.py
- poker44/base/neuron.py
- poker44/miner_heuristics.py
- poker44/utils/config.py
- poker44/utils/misc.py
- poker44/utils/model_manifest.py
- poker44/validator/synapse.py
