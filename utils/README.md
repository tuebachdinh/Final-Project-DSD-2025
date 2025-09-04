## Utility Functions Documentation

### Signal Processing Utilities (`utils/signal_processing/`)

#### `PulseAnalyse.m` - Comprehensive Pulse Wave Analysis
- **Purpose**: Advanced fiducial point detection and pulse wave index calculation
- **Features**: Multi-derivative analysis, quality assessment, beat detection
- **Outputs**: 40+ pulse wave indices including AI, RI, SI, AGI variants

#### `PulseAnalyse10.m` - Streamlined Pulse Analysis  
- **Purpose**: Focused cardiovascular index extraction
- **Features**: Automated fiducial point detection, quality filtering
- **Outputs**: Core CV indices with median aggregation

#### `extractFreqMorphFeatures.m` - Feature Engineering
- **Purpose**: Extracts frequency domain and morphological features
- **Features**: Spectral analysis, harmonic ratios, statistical moments
- **Outputs**: 15+ features including energy bands, time-to-peak, rise/decay times

#### `fitTwoGaussiansPPG.m` - Peak Detection
- **Purpose**: Robust P1/P2 peak detection using Gaussian mixture models
- **Features**: Constrained optimization, goodness-of-fit metrics
- **Outputs**: Peak locations, Gaussian parameters, fit quality (R², RMSE)

#### `computeSimpleSQI.m` - Signal Quality Assessment
- **Purpose**: Template-based signal quality evaluation
- **Features**: Correlation-based quality index with adaptive thresholding
- **Outputs**: SQI score (0-1) and binary quality classification

### Deep Learning Utilities (`utils/deep_learning/`)

#### `model_interpretability.m` - Model Explanation Engine
- **Purpose**: Comprehensive model interpretability analysis
- **Methods**: Occlusion analysis, perturbation sensitivity, attention visualization
- **Features**: Multi-channel support, statistical aggregation, visualization

#### `plot_importance_overlay.m` - Importance Visualization
- **Purpose**: Overlays importance scores on original waveforms
- **Features**: Temporal highlighting, multi-channel support, customizable thresholds
- **Outputs**: Publication-ready importance plots with sensitivity overlays

### General Utilities (`utils/others/`)

#### `save_figure.m` - Consistent Figure Management
- **Purpose**: Standardized figure saving with automatic directory creation
- **Features**: Part-based naming, PNG format, project root detection
- **Usage**: `save_figure('figure_name', part_number)`

#### `summarize_metrics_create_table.m` - Results Aggregation
- **Purpose**: Creates comprehensive performance tables from model results
- **Features**: Multi-model comparison, CSV export, statistical formatting
- **Outputs**: 9×4 tables (3 models × 3 input types × 4 metrics)

#### `make_stacked_plots.m` - Performance Visualization
- **Purpose**: Creates grouped bar plots with numeric labels
- **Features**: Clean/augmented comparison, multiple metrics, publication quality
- **Outputs**: 8 grouped bar plots (2 datasets × 4 metrics)

#### `count_params.m` - Model Complexity Analysis
- **Purpose**: Counts learnable parameters across different network types
- **Features**: Supports SeriesNetwork, DAGNetwork, dlnetwork
- **Usage**: Model complexity comparison and computational analysis