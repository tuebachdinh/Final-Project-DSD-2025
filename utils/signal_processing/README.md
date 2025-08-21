# Signal Processing Algorithms
This folder contains the main algorithms for pulse wave and cardiovascular signal analysis, including feature extraction, fiducial point detection, and clinical index calculation.

---

## 1. PulseAnalyse10 Function Summary

### Overview

**`PulseAnalyse10`** is a MATLAB function for automated extraction of cardiovascular (CV) indices from pulsatile signals such as PPG or arterial pressure waveforms. It identifies key fiducial points (peaks, notches, inflections) and calculates various clinical indices, supporting both single-pulse and multi-pulse signals.

---

### Main Benefits

- **Comprehensive:** Extracts a wide range of CV indices (AI, SI, RI, timing/amplitude ratios, etc.).
- **Flexible:** Handles both single and multiple beats.
- **Robust Landmark Detection:** Uses derivatives and adaptive methods for fiducial point finding.
- **Signal Quality Assessment:** Flags and excludes low-quality pulses.
- **Visualization:** Includes plotting for diagnostics and review.
- **Modular:** Structured for easy customization and extension.

---

### File and Function Location

| Name                | Location/Line        | Purpose                                    |
|---------------------|---------------------|--------------------------------------------|
| `PulseAnalyse10.m`  | line 1              | Main function                              |
| `setup_options`     | ~line 63            | Initializes/checks analysis options        |
| `setup_up`          | ~line 87            | Sets parameters and filter settings        |
| `determine_no_of_pulses` | ~line 112      | Single vs. multi-pulse detection           |
| `elim_vlfs/vhfs/vhfs2`   | ~lines 147,183,332 | Preprocess: Remove low/high freq noise |
| `adaptPulseSegment` | ~line 247           | Adaptive beat segmentation, onset/peak     |
| `assess_signal_quality` | ~line 529        | Signal quality evaluation                  |
| `calc_derivs2, savitzky_golay` | ~lines 649, 677 | Derivative computation            |
| `identify_fiducial_pts` | ~line 769        | Finds fiducial points (all peaks, etc.)    |
| `identify_p2`       | ~line 1125          | Detects second peak (reflected wave, P2)   |
| `calc_stiffness_inds` | ~line 1247         | Computes all CV indices from fiducials     |
| `plot_fiducial_points` | ~line 1441        | Diagnostic plot of fiducial points         |
| `plot_cv_inds`      | ~line 1643           | Diagnostic plot of indices                 |

---

### Typical Usage Example

```matlab
S.v = ppg_signal(:);    % Input signal
S.fs = fs;              % Sampling rate
options.do_plot = true; % Plot results
[cv_inds, fid_pts, pulses, S_filt] = PulseAnalyse10(S, options);
```
---

### Key Outputs:
- cv_inds: Structure with calculated CV indices (AI, SI, etc.).
- fid_pts: Structure with indices of fiducial points (systolic peak, notch, P1, P2, etc.).
- pulses: Struct with segmentation and signal quality.
- sigs/S_filt: Filtered signal.

### Limitation
1. **Merged or Absent Peaks**
   - AThe code struggles when the second peak (P2) is merged into the first or is only a shoulder—sometimes returning no value or an inaccurate estimate.
   - No parametric (e.g., Gaussian fitting) or machine learning solution for these cases by default.

2. **Sensitivity to Noise and Motion Artifacts**
   - The function relies on signal derivatives and simple adaptive thresholds, which can be disrupted by motion artifacts, baseline drift, and noise—common in wearable data.
   - No integration of accelerometer or signal quality index (SQI) for artifact rejection.

3. **Single-Channel, Beat-by-Beat Analysis**
   - Does not natively combine PPG and ECG for multimodal features (e.g., pulse arrival/transit time).
   - Limited to analyzing one signal at a time.

4. **Limited Real-Time or Continuous Monitoring Capability**
   - Primarily designed for offline or batch processing; not optimized for efficient real-time use or resource-limited embedded systems.

5. **Lack of Advanced Morphological or Machine Learning Features**
   - Only classic time-domain and derivative-based indices are provided; no use of frequency-domain, shape descriptors, or ML-extracted features.
   - No deep learning or data-driven pattern recognition.

6. **Manual Review Still Needed for Edge Cases**
   - Ambiguous or pathological signals may require human review or manual correction.

---

## 2. Gaussian Fitting for Pulse Wave Second Peak (P2)

### Function File: **`fitTwoGaussiansPPG.m`**

### Theoretical Basis
- Problem: Second peak (P2, reflected wave) may be only a shoulder or merged with main peak.
- Solution: Model the waveform as the sum of two Gaussians:
    + One for main systolic (P1)
    + One for reflected (P2), even if only a shoulder
### Steps
1. Fit the pulse as the sum of two Gaussian functions using nonlinear least squares.
2. Initialize parameters (amplitude, center, width) based on the waveform.
3. Optimize parameters to minimize error between sum of Gaussians and signal.
4. Identify P1 and P2 as maxima of the respective Gaussians.

#### Benefits
- Reveals P2 position even if no obvious local max exists.
- Works with noisy/ambiguous/aged pulse waveforms.

### Typical Usage Example

```matlab
idx = 1; % example subject index
sig = waves.PPG_Radial(idx,:); % or your single-beat signal
[fitCurve, params, p1, p2] = fitTwoGaussiansPPG(sig);
```
---
## 3. Frequency Domain and Morphology Features

### Function File: **`extractFreqMorphFeatures.m`**

### Theoretical Basis
- The shape of the pulse wave contains diagnostic information not only in time, but also in frequency and overall morphology.
- Vascular stiffness, reflected waves, and disease can affect the harmonic content and statistical shape descriptors.

### Example Features
- **Spectral Energy:** Proportion of total signal power in various frequency bands (Fourier or wavelet analysis).
- **Dominant Frequency:** Frequency with highest spectral energy.
- **Harmonic Ratios:** Ratios between energy in first/second/third harmonics.
- **Morphological Descriptors:** Skewness, kurtosis, area under the curve, time-to-peak, rise/decay time, etc.

### Steps
1. Apply FFT or wavelet transform to each pulse/beat.
2. Extract spectral features and ratios.
3. Calculate time-domain morphology features.
4. Use these features for vascular aging classification, blood pressure estimation, or artifact detection.

#### Benefits
- Detects subtle changes not visible in classic fiducial points.
- Adds robustness for ML models and multi-feature analysis.

---

## 4. Signal Quality Indices (SQI) and Artifact Rejection

### Function File: **`computeSimpleSQI.m`**

### Theoretical Basis
- Wearable signals (PPG, ECG) are prone to noise, baseline drift, and motion artifacts.
- Signal Quality Indices (SQI) score each beat/segment for reliability and suitability for analysis.

### Example Techniques
- **Correlation-Based SQI:** Compare each beat to a template; low correlation suggests artifact.
- **Saturation/Clipping Detection:** Identify signal plateaus or extremes.
- **Noise/Variance Metrics:** High-frequency noise or excessive baseline wander indicates poor quality.
- **Accelerometer Fusion:** Combine SQI with motion sensor data.

### Steps
1. For each detected beat/segment, compute one or more SQI metrics.
2. Set thresholds (data-driven or empirical) for "acceptable" vs "artifact."
3. Exclude, flag, or downweight beats with poor SQI in further analysis.
4. Optionally, provide real-time feedback for re-measurement or repositioning.

#### Benefits
- Prevents erroneous analysis on low-quality data.
- Improves the reliability of clinical indices and any ML models.
- Enables robust monitoring in real-world, ambulatory environments.

---
## 5. (Optional) Machine Learning and Multimodal Fusion

### Theoretical Basis
- Combining multiple features (classic, spectral, morphological, SQI, and multimodal) in ML models enables more accurate and robust prediction of vascular health, blood pressure, and artifact detection.

### Example Techniques
- **Classical ML:** Random Forests, SVM, Gradient Boosted Trees using engineered features.
- **Neural Networks:** 1D CNNs or LSTMs for end-to-end learning on raw/filtered waveforms.
- **Multimodal Fusion:** Combine features from PPG, ECG, and accelerometer for richer models.
- **On-device/real-time optimization:** Lightweight models for embedded applications.

### Steps
1. Aggregate features from time, frequency, morphology, and SQI.
2. Train/test ML models using labeled data (clinical ground truth, artifact/no artifact, etc.).
3. Integrate model inference into the signal processing pipeline.

#### Benefits
- Exploits all available data and signal modalities.
- Boosts accuracy for complex tasks (e.g., cuffless BP, arrhythmia detection, robust PAT/PTT estimation).
- Reduces the need for manual review or hand-crafted rules.

---

