# System Architecture

## Architecture Diagram

```mermaid
flowchart TD
    In[User Task / Query] --> FE[Feature Extractor]
    FE --> CG{Confidence Gate}
    
    CG -- "High Confidence" --> Direct[Low-Cost / Direct Agent]
    CG -- "Moderate Uncertainty" --> Fallback[High-Capacity Model / Deep Search]
    CG -- "High Uncertainty" --> RAG[RAG Retrieval Augmented Agent]
    
    Direct --> BC{Budget & Context Check}
    Fallback --> BC
    RAG --> BC
    
    BC -- "Within Limit" --> Next[Next Step / State Update]
    BC -- "Budget Exceeded" --> Synth[Final Synthesis & Termination]
    Next --> Done[Output Answer]
    Synth --> Done
```

## Core Subsystems
1. **Feature Extractor (`src/features/extractor.py`)**: Computes token entropy, likelihood stats, and step token usage.
2. **Confidence Gate (`src/gating/confidence_gate.py`)**: Applies calibrated criteria to route execution.
3. **MAR Coordinator (`src/mar/system.py`)**: Controls state transitions and interactions between agents.
4. **Context Budget Tracker (`src/evaluation/budget.py`)**: Ensures compute and token boundaries are respected.
