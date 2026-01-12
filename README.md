# Empirical Scaling Harness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/empirical-scaling-harness/blob/main/scaling_experiment.ipynb)

> **One-click experiment**: Click the badge above to run the complete scaling law experiment in Google Colab. No setup required - just click and run all cells.

Automated verification of power-law scaling for transformer architecture variants.

## Scientific Question

**Hypothesis**: Does the SwiGLU activation function shift the scaling exponent α, or does it merely provide a constant offset in compute efficiency compared to GeLU?

## Quick Start

### Option A: Google Colab (Recommended)

1. Click the "Open in Colab" badge at the top of this README
2. In Colab: Runtime → Run all
3. Wait 2-8 hours depending on configuration
4. Download the `scaling_plot.png` and results when complete

**Runtime estimates:**
- Anchors only (3M, 10M, 30M): ~2-3 hours
- Full sweep (includes 85M holdout): ~6-8 hours

### Option B: Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate configuration (dry run)
python main.py --dry-run

# 3. Run experiments
python main.py                    # Full sweep
python main.py --anchors-only     # Faster (skip 85M holdout)

# 4. Analyze results
jupyter notebook analysis.ipynb
```

## Experiment Grid

| Model | Params | Tokens (D) | Compute (C=6ND) | Purpose |
|-------|--------|------------|-----------------|---------|
| Anchor 1 | 3M | ~60M | ~1.1e15 | Fit power law |
| Anchor 2 | 10M | ~200M | ~1.2e16 | Fit power law |
| Anchor 3 | 30M | ~600M | ~1.1e17 | Fit power law |
| Holdout | 85M | ~1.7B | ~8.7e17 | Validate prediction |

## Project Structure

```
empirical-scaling-harness/
├── scaling_experiment.ipynb  # Self-contained Colab notebook (run this!)
├── config.py                 # Model config builder for target param counts
├── model.py                  # Transformer with configurable activation (GeLU/SwiGLU)
├── data.py                   # TinyStories dataset loading
├── trainer.py                # Training loop with CSV + TensorBoard logging
├── sweep.py                  # Individual experiment runner CLI
├── main.py                   # Master orchestrator for full sweep
├── analysis.ipynb            # Standalone analysis notebook
├── requirements.txt          # Python dependencies
└── logs/                     # Experiment outputs (gitignored)
```

## Individual Experiment

Run a single experiment:

```bash
# Train a 3M parameter model with GeLU
python sweep.py --params 3000000 --activation gelu

# Train a 10M parameter model with SwiGLU
python sweep.py --params 10000000 --activation swiglu

# Dry run to check configuration
python sweep.py --params 3000000 --activation gelu --dry-run
```

## Key Design Choices

### Model Architecture
- **Decoder-only transformer** with causal attention
- **Pre-norm architecture** (LayerNorm before attention/FFN)
- **Weight-tied embeddings** (input and output share weights)
- **SwiGLU FFN**: Uses `d_ff = 8/3 × d_model` to match GeLU param count

### Training
- **Dataset**: TinyStories (fast convergence, good for scaling studies)
- **Tokenizer**: GPT-2 (vocab_size=50257)
- **Optimizer**: AdamW with cosine LR schedule
- **Chinchilla ratio**: 20 tokens per parameter

### Hardware Optimization (Google Colab)
- Mixed precision (fp16) for larger models
- Gradient accumulation for effective larger batches
- Conservative batch sizes to fit in ~15GB VRAM

## Analysis Output

The `analysis.ipynb` notebook produces:

1. **Power law fits** for each activation:
   ```
   L(C) = a × C^(-b)
   ```

2. **Holdout validation**: Predicted vs actual loss for 85M models

3. **Scaling plot**: Log-log plot comparing both activations

4. **Conclusion**: Whether SwiGLU shifts the exponent or just the coefficient

## Expected Results

If the hypothesis is correct (SwiGLU provides constant offset):
- Scaling exponents (b) should be similar for both activations
- SwiGLU coefficient (a) should be lower (better efficiency)
- Prediction error on holdout should be < 5%

## Monitoring

TensorBoard logs are saved during training:

```bash
tensorboard --logdir logs/
```

## Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev,notebook]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .
```

## Citation

This project replicates methodology from:
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
