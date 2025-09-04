# Research Workflow and Reasoning Report

## Project Context

**Research Team:**
- **Student**: Dinh Bach Tue (Aalto University)
- **Advisor**: PhD Researcher Bahmani Nima
- **Supervisor**: Professor Stephan Sigg
- **Institution**: Aalto University, Finland

**Research Objective:** Create a robust foundation for transferring knowledge from synthetic waveform data to real wearable device data, focusing on radial artery (wrist) measurements for non-invasive cardiovascular health monitoring.

---

## Research Evolution and Dataset Transition

### Initial Exploration (2015 Dataset - Archived)
The project began with the **HaeMod dataset (2015)** containing pressure, flow, and area waveforms but **no PPG data**. Basic signal processing was applied to extract features from pressure waveforms for PWV prediction using classical ML models. This work is preserved in the `archieved/` folder.

**Key Limitation:** Lack of PPG signals made it unsuitable for wearable device applications.

### Main Analysis (2019 Dataset - Current Focus)
Transitioned to the **PWDB dataset (2019)** which includes **PPG waveforms** essential for wearable device applications. The comprehensive analysis focuses on **radial artery (wrist) measurements** as the target for real-world deployment.

**Strategic Decision:** PPG availability enables direct translation to consumer wearable devices.

---

## Detailed Research Workflow

### Phase 1: Exploratory Analysis (Parts 1-3)

**Part 1-3**: Data preparation, waveform visualization by stiffness/age groups, and PTT analysis

**Key Findings:**
- Distinct waveform patterns across stiffness groups with physiological explanations
- PTT analysis essential for real-device deployment where PTT measurements enable PWV calculation

**Research Reasoning:** Understanding fundamental signal characteristics before applying complex models ensures physiologically meaningful results.

### Phase 2: Feature-Based, Classical Approaches (Parts 4-6)

**Part 4-6**: Classical regression, ML on raw waveforms, and advanced signal processing

**Key Insights:**
- Signal Processing: Latest PulseAnalyse toolkit for comprehensive feature extraction
- Challenge: Real devices struggle with feature extraction due to noise

**Research Reasoning:** Establishing baseline performance with traditional methods provides comparison benchmark for deep learning approaches.

### Phase 3: Deep Learning Solutions (Parts 7-9)

**Part 7-9**: Data augmentation, classical ML with augmented data, and deep neural networks

**Key Innovations:**
- Augmentation improves robustness AND model attention to correct waveform regions
- Architecture Comparison: CNN vs GRU vs **TCN** (best performer)
- Input Modalities: PPG-only, Area-only, **Both (optimal)**

**Research Reasoning:** Systematic architecture comparison ensures optimal model selection, while augmentation addresses real-world deployment challenges.

### Phase 4: Model Understanding & Validation (Parts 10-12)

**Part 10-12**: Model interpretability, cross-validation, and final evaluation

**Key Results:**
- Interpretability: Occlusion and perturbation analysis reveal model focus areas
- Final Model: TCN with PPG+Area input, trained on clean+augmented data, 5-fold CV

**Research Reasoning:** Model interpretability ensures clinical relevance and builds trust for medical applications.

---

## Key Research Findings

### 1. Data Augmentation Benefits
- **Robustness**: Improved generalization to noisy real-world signals
- **Attention Correction**: Models focus on physiologically relevant waveform regions
- **Performance**: Maintains accuracy while increasing deployment readiness

**Clinical Significance:** Addresses the primary challenge of synthetic-to-real domain transfer.

### 2. Optimal Model Configuration
- **Architecture**: **TCN (Temporal Convolutional Network)** outperforms CNN and GRU
- **Input Channels**: **PPG + Area** superior to single-channel approaches
- **Training Strategy**: **Mixed clean + augmented data** for best performance

**Technical Innovation:** First systematic comparison of temporal architectures for pulse wave analysis.

### 3. Model Interpretability Insights
- **Critical Regions**: Models focus on systolic upstroke and dicrotic notch
- **Channel Complementarity**: PPG and Area provide different but complementary information
- **Physiological Relevance**: Attention patterns align with cardiovascular physiology

**Clinical Validation:** Model attention aligns with known cardiovascular biomarkers.

---

## Research Motivation and Clinical Impact

### Clinical Need
- **Problem**: Arterial stiffness (cfPWV) predicts cardiovascular events but requires expensive clinical equipment
- **Opportunity**: Consumer devices can capture pulse signals but lack validated algorithms for health assessment

### Technical Challenge
- **AI Gap**: Advanced ML/DL can extract physiological insights from noisy, real-world wearable data
- **Interpretability**: Clinical adoption requires understanding what models learn from pulse morphology
- **Transfer Learning**: Bridging the gap between clean simulated data and noisy real-world signals

### Research Solution
- **Comprehensive Pipeline**: End-to-end analysis from data preparation to model deployment
- **Augmentation Strategy**: Novel approach improving both robustness and interpretability
- **Multi-Modal Integration**: Optimal combination of available wearable sensor data

---

## Future Research Directions

### Phase 2: Wearable Device Deployment
- **Transfer Learning**: Adapt synthetic-trained models to real wrist-worn devices
- **Domain Adaptation**: Bridge simulation-to-reality gap using established baseline
- **PTT Integration**: Leverage PTT analysis for device-based PWV calculation
- **Robustness Validation**: Test augmentation strategies on real noisy signals
- **Clinical Translation**: Validate against gold-standard PWV measurements

### Technical Contributions to Field
- **Comprehensive Baseline**: Systematic evaluation of synthetic data for arterial stiffness prediction
- **Augmentation Strategy**: Novel approach improving both robustness and model interpretability
- **Architecture Comparison**: First systematic comparison of CNN/GRU/TCN for pulse wave analysis
- **Multi-Modal Integration**: Optimal combination of PPG and Area signals for wrist measurements
- **Transfer-Ready Pipeline**: Foundation for synthetic-to-real domain adaptation

---

## Research Impact and Significance

This work establishes a **comprehensive foundation** for synthetic-to-real transfer learning in cardiovascular monitoring, addressing the critical gap between laboratory research and real-world wearable device deployment. The systematic approach, from basic signal analysis to advanced deep learning with interpretability, provides a **replicable methodology** for similar biomedical signal processing challenges.

**Key Innovation:** The discovery that data augmentation not only improves robustness but also corrects model attention to physiologically relevant regions represents a significant contribution to both machine learning and biomedical engineering fields.

**Clinical Relevance:** The focus on wrist-based measurements and PPG signals directly addresses the practical constraints of consumer wearable devices, making this research immediately applicable to real-world health monitoring applications.