# Testing Guide

The project relies on `pytest` to guarantee preprocessing and training logic integrity without having to run 1000-epoch
experiments blindly.

## Running Tests

Run the entire suite locally:

```bash
pytest
```

To run with coverage:

```bash
pytest --cov=src/overfitlab
```

## Coverage Scope

1. **Data Tests**: Verifies deterministic splitting, tensor conversion, and dimensions.
2. **Model Tests**: Ensures forward passes execute cleanly and models maintain expected architectural shapes.
3. **Training Tests**: Performs a rapid 2-epoch test on dummy data to ensure the training loop logs metrics correctly
   without throwing exceptions.