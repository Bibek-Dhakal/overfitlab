# Usage Guide

## Running the Experiment

The primary entry point to this project is the `experiment.py` script. It handles generating data, instantiating the
models, training both loops, and producing visual results.

```bash
python -m overfitlab.experiment
```

### Outputs

- **Console Logs**: Prints the loss metrics for both training and validation phases per 100 epochs.
- **`experiment_results.png`**: Generates and saves a two-pane matplotlib chart visually proving the train/validation
  loss divergence in the baseline model, and the corrected curve in the regularized model.

## Notebook Usage

If you prefer Jupyter Notebooks, open `notebooks/OverfitLab_Experiment.ipynb` (which is standard Jupyter JSON) in your
preferred environment (e.g., VS Code or JupyterLab).