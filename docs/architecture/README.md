# System Architecture

## Component Flow

```mermaid
graph TD
    A[raw dataset generation] --> B[deterministic train/val split]
    B --> C[Baseline MLP Setup]
    B --> D[Regularized MLP Setup]
    
    C --> E[Baseline Training Loop]
    D --> F[Regularized Training Loop]
    
    E --> G[Overfit Metrics Logged]
    F --> H[Corrected Metrics Logged]
    
    G --> I[Evaluation & Plotting]
    H --> I
    
    I --> J[experiment_results.png]
```

## Directory Responsibilities

- `src/overfitlab/data.py`: Handles pure, deterministic data creation and tensor conversion.
- `src/overfitlab/model.py`: Neural network class definitions.
- `src/overfitlab/train.py`: Agnostic training loops that accept models and output tracking dictionaries.
- `src/overfitlab/experiment.py`: The orchestrator joining data, models, and plotting.
