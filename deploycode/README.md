# CarbonSONet: Physics-Informed Carbon Intensity Forecasting

Code repository for reproducing the results in the paper.

## Directory Structure

- models/ - Model architecture (carbon_sonet_v24.py)
- experiments/ - Training and evaluation scripts
  - un_ablation_v24_final3.py - Main training script for full model and ablation variants
- igures/ - Figure generation scripts
  - generate_fig3_v3.py - Generate Figure 3 (prediction examples)
  - generate_comparison_figures_v5.py - Generate comparison figures
  - plot_comparison_v6.py - Generate 16-country comparison figures
- un_multiseed.py - Multi-seed evaluation script

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with 32GB+ memory

## Training

To reproduce the full model results across all 16 European countries:

`ash
cd /home/ubuntu/Carbon_intensity_forecasting
python3 experiments/run_ablation_v24_final3.py
`

The script will train the full model and ablation variants across all 16 European countries.

## Multi-Seed Evaluation

To evaluate seed sensitivity:

`ash
python3 run_multiseed.py
`

## Figures

To reproduce Figure 3 (prediction examples):

`ash
python3 figures/generate_fig3_v3.py
`

## Data

Preprocessed datasets for all 16 countries are available in the deploydata/ directory.
