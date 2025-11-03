```mermaid
graph TD
    %% Data Pipeline
    A[Raw MRI Data] --> B[Data Collection]
    B --> |4 Modalities| C[Preprocessing]
    C --> |Brain Masking| D[Normalized Data]
    D --> |Patch Generation| E[Training Data]

    %% Model Pipeline
    E --> F[3D U-Net Model]
    F --> |Encoder Path| G[Feature Extraction]
    G --> |Bottleneck| H[Feature Processing]
    H --> |Decoder Path| I[Segmentation Output]

    %% Post-Processing & Analysis
    I --> J[Results Processing]
    J --> K[Segmentation Maps]
    J --> L[Uncertainty Maps]
    J --> M[Model Explainability]
    J --> N[3D Visualization]

    %% Backend Integration
    O[FastAPI Server] --> |Input| P[ONNX Runtime]
    P --> |Processing| Q[Inference Results]
    Q --> |Output| R[API Response]

    %% Frontend Components
    R --> S[React UI]
    S --> |Display| T[Interactive Viewer]
    T --> |Components| U[3D Renderer]
    T --> |Components| V[Uncertainty View]
    T --> |Components| W[Explainability View]

    %% Data Flow Details
    subgraph Input Data
        A --> |T1| B
        A --> |T1ce| B
        A --> |T2| B
        A --> |FLAIR| B
    end

    subgraph Model Architecture
        F --> |Attention Gates| G
        H --> |Skip Connections| I
    end

    subgraph Evaluation Metrics
        I --> |Dice Score| X[Performance Metrics]
        I --> |Cross-Entropy| X
        I --> |Accuracy| X
    end

    %% Status Indicators
    style A fill:#90EE90 %% Implemented
    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
    style F fill:#90EE90
    style G fill:#90EE90
    style H fill:#90EE90
    style I fill:#90EE90
    style J fill:#90EE90
    style K fill:#FFB6C1 %% Planned
    style L fill:#FFB6C1
    style M fill:#FFB6C1
    style N fill:#FFB6C1
    style O fill:#90EE90
    style P fill:#90EE90
    style Q fill:#90EE90
    style R fill:#90EE90
    style S fill:#FFB6C1
    style T fill:#FFB6C1
    style U fill:#FFB6C1
    style V fill:#FFB6C1
    style W fill:#FFB6C1
    style X fill:#90EE90

classDef implemented fill:#90EE90;
classDef planned fill:#FFB6C1;
```

### Legend
- 🟢 Green: Implemented Features
- 🔴 Red: Planned Features

### Implementation Status
1. **Implemented (🟢)**
   - Data Collection & Preprocessing
   - Model Architecture & Training
   - Basic Backend Integration
   - Performance Metrics
   - ONNX Export

2. **Planned (🔴)**
   - Segmentation Visualization
   - Uncertainty Quantification
   - Model Explainability
   - 3D Rendering
   - Interactive Frontend

### Next Steps
1. Implement visualization components
2. Add uncertainty estimation
3. Develop explainability features
4. Create 3D renderer
5. Complete frontend integration