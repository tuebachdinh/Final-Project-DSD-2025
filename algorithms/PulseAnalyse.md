# PulseAnalyse.m - Comprehensive Pulse Wave Analysis Toolbox

## Overview

PulseAnalyse is a research-grade MATLAB function for extracting pulse wave indices from pulsatile signals (PPG, arterial pressure). It provides comprehensive analysis of cardiovascular waveforms with advanced fiducial point detection, quality assessment, and feature extraction capabilities.

**Author**: Peter H. Charlton, University of Cambridge  
**Version**: 1.3beta  
**License**: GNU Public License

## Core Functionality

### Input Requirements
```matlab
S = struct();
S.v = signal_data;    % Pulse wave amplitudes (vector)
S.fs = 125;           % Sampling frequency (Hz)
S.ht = 1.75;          % Subject height in meters (optional)

[pw_inds, fid_pts, pulses, sigs] = PulseAnalyse(S, options);
```

### Main Outputs
- **`pw_inds`**: 39+ pulse wave indices (AI, RI, SI, etc.)
- **`fid_pts`**: Fiducial point locations and amplitudes
- **`pulses`**: Beat detection and quality assessment results
- **`sigs`**: Original and processed signals with derivatives

## Processing Pipeline

### 1. **Signal Preprocessing**
```
Input Signal → Downsampling → Beat Detection → Quality Assessment → Filtering
```

**Beat Detection Algorithms:**
- **IMS** (Incremental Merge Segmentation) - Default
- **AMPD** (Automatic Multiscale Peak Detection)
- **Aboy** (Adaptive threshold method)
- **Bishop** (Scalogram-based detection)
- **qPPG** (PhysioNet implementation)

**Quality Assessment:**
- Cross-correlation with template beats (threshold: 0.95)
- 10-second windowed analysis
- Excludes low-quality beats from median calculations

### 2. **Signal Filtering**
```
VHF Elimination → VLF Elimination → Baseline Correction
```

**Filter Specifications:**
- **High-frequency cutoff**: 20 Hz (Fpass), 15 Hz (Fstop)
- **Low-frequency cutoff**: 0.8 Hz (Fpass), 0.6 Hz (Fstop)
- **Method**: Kaiser window FIR filters with Tukey windowing

### 3. **Derivative Calculation**
```
Signal → 1st Derivative → 2nd Derivative → 3rd Derivative
```

**Method**: Savitzky-Golay filters (9-point window)
- Handles edge effects by pulse repetition for single beats
- Provides robust derivative estimation for noisy signals

### 4. **Fiducial Point Detection**

#### **17 Fiducial Points Detected:**

| Point | Location | Detection Method |
|-------|----------|------------------|
| **f1, f2** | Pulse onset/end | Beat detection boundaries |
| **s** | Systolic peak | Maximum amplitude |
| **ms** | Maximum slope | Peak of 1st derivative |
| **a** | Early systolic peak | Peak in 2nd derivative before ms |
| **b** | Systolic minimum | Minimum in 2nd derivative after a |
| **c** | Mid-systolic peak | Peak in 2nd derivative between b and e |
| **d** | Late systolic minimum | Minimum in 2nd derivative between c and e |
| **e** | Dicrotic peak | Peak in 2nd derivative (60% of cycle) |
| **f** | Late diastolic minimum | Minimum in 2nd derivative after e |
| **dic** | Dicrotic notch | Same as e |
| **dia** | Diastolic peak | Peak in signal after dic |
| **p1pk, p1in** | First peak | Systolic peak (original/inflection) |
| **p2pk, p2in** | Second peak | Diastolic peak (original/inflection) |
| **ms2** | Secondary max slope | Peak in 1st derivative between p1 and p2 |

#### **Advanced P2 Detection Algorithm:**
```matlab
% Multi-strategy approach for merged peaks in elderly subjects
1. Third derivative minima before dicrotic notch
2. Fallback to post-dicrotic minima if P2 < P1
3. Signal peak validation between estimates
4. Peak enhancement for merged waveforms
```

### 5. **Pulse Wave Indices Calculation**

#### **Timing Indices (12)**
- **T**: Pulse duration
- **IHR**: Instantaneous heart rate
- **CT**: Crest time (onset to systolic peak)
- **delta_t**: Time between systolic and diastolic peaks
- **SI**: Stiffness Index = height/delta_t
- **t_sys, t_dia**: Systolic/diastolic durations
- **t_ratio**: Systolic/diastolic ratio
- **prop_s**: Proportion of systolic upslope
- **t_p1_dia, t_p2_dia**: P1/P2 to diastolic peak times
- **t_b_c, t_b_d**: Fiducial point intervals

#### **Amplitude Indices (8)**
- **AI**: Augmentation Index = 100×(P2-P1)/pulse_amplitude
- **AP**: Augmentation Pressure = P2-P1
- **RI**: Reflection Index = diastolic_peak/systolic_peak
- **RI_p1, RI_p2**: Reflection indices using P1/P2
- **ratio_p2_p1**: P2/P1 amplitude ratio
- **pulse_amp**: Total pulse amplitude
- **dia_amp**: Diastolic peak amplitude

#### **Area Indices (3)**
- **A1**: Systolic area (onset to dicrotic notch)
- **A2**: Diastolic area (dicrotic notch to end)
- **IPA**: Inflection Point Area = A2/A1

#### **Derivative Indices (16)**
- **ms**: Maximum slope value
- **ms_div_amp**: Maximum slope normalized by amplitude
- **a_div_amp** through **e_div_amp**: 2nd derivative peaks normalized
- **b_div_a** through **e_div_a**: 2nd derivative ratios
- **AGI**: Aging Index = (b-c-d-e)/a
- **AGI_inf**: Informal Aging Index = (b-e)/a
- **AGI_mod**: Modified Aging Index = (b-c-d)/a
- **slope_b_c, slope_b_d**: Normalized slopes between fiducial points

#### **Combined Indices (2)**
- **IPAD**: IPA + d/a (combined area and derivative)
- **k**: Stiffness constant = s/((s-ms)/pulse_amp)

## Advanced Features

### **1. Multi-Signal Support**
- **PPG signals**: Optimized for photoplethysmography
- **Pressure signals**: Arterial blood pressure waveforms
- **Single/Multiple beats**: Automatic detection and processing

### **2. Quality Control**
```matlab
% Correlation-based quality assessment
template = mean(high_quality_beats);
for each_beat
    correlation = corr2(beat, template);
    quality(beat) = correlation > 0.95;
end
```

### **3. Robust Processing**
- **Noise handling**: Multiple filtering stages
- **Edge effects**: Pulse repetition and windowing
- **Missing points**: Graceful degradation with NaN handling
- **Buffer zones**: Physiological constraints on search regions

### **4. Visualization**
- **Fiducial point plots**: All detected points overlaid
- **Derivative plots**: 1st, 2nd, 3rd derivatives with annotations
- **Index illustrations**: Visual representation of calculated indices
- **Quality assessment plots**: Beat-by-beat quality visualization

## Configuration Options

### **Essential Options**
```matlab
options.do_filter = 1;              % Enable filtering (recommended)
options.do_quality = 1;             % Enable quality assessment
options.normalise_pw = 1;           % Normalize to 0-1 range
options.beat_detector = 'IMS';      % Beat detection algorithm
options.exclude_low_quality_data = 1; % Use only high-quality beats
```

### **Visualization Options**
```matlab
options.do_plot = 1;                % Generate plots
options.plot_third_deriv = 1;       % Include 3rd derivative
options.plot_areas = 1;             % Show systolic/diastolic areas
options.save_folder = './results/'; % Save location
```

### **Advanced Options**
```matlab
options.downsample = 1;             % Enable downsampling
options.downsample_freq = 125;      % Target frequency
options.calibrate_pw = 1;           % Pressure calibration
options.calc_average_pw = 1;        % Calculate average waveform
```

## Computational Methods

### **Savitzky-Golay Derivatives**
```matlab
% 1st derivative coefficients (9-point)
coeffs = [-4, -3, -2, -1, 0, 1, 2, 3, 4] / 60;
```

### **Kaiser Window Filtering**
```matlab
% Adaptive filter design
[N, Wn, BETA, TYPE] = kaiserord([Fstop, Fpass]/(fs/2), [1, 0], [Dstop, Dpass]);
filter_coeffs = fir1(N, Wn, TYPE, kaiser(N+1, BETA), 'scale');
```

### **Peak Detection Algorithm**
```matlab
% Find peaks and troughs
peaks = find(sig(2:end-1) > sig(1:end-2) & sig(2:end-1) > sig(3:end));
troughs = find(sig(2:end-1) < sig(1:end-2) & sig(2:end-1) < sig(3:end));
```

## Clinical Applications

### **Arterial Stiffness Assessment**
- **PWV estimation**: Using timing-based indices (SI, CT)
- **Wave reflection**: Augmentation indices (AI, RI)
- **Aging assessment**: AGI and derivative ratios

### **Cardiovascular Risk**
- **Pulse morphology**: Shape-based indices
- **Hemodynamic load**: Area and amplitude ratios
- **Vascular compliance**: Combined timing and amplitude features

## Performance Characteristics

### **Computational Complexity**
- **Single beat**: ~50ms processing time
- **Multiple beats**: Linear scaling with beat count
- **Memory usage**: ~10MB for 1000 beats

## Usage Examples

### **Basic Analysis**
```matlab
% Load PPG signal
load('ppg_data.mat');
S.v = ppg_signal;
S.fs = 125;

% Run analysis
[pw_inds, fid_pts, pulses, sigs] = PulseAnalyse(S);

% Access results
AI = pw_inds.AI.v;          % Augmentation Index
RI = pw_inds.RI.v;          % Reflection Index
p2_location = fid_pts.ind.p2pk; % P2 peak index
```

### **Batch Processing**
```matlab
for subject = 1:num_subjects
    S.v = ppg_data(subject,:);
    S.fs = 125;
    S.ht = heights(subject);
    
    [pw_inds(subject), ~, ~, ~] = PulseAnalyse(S, options);
end
```

## Dependencies

- **MATLAB Signal Processing Toolbox**
- **Statistics and Machine Learning Toolbox** (for quality assessment)
- **No external libraries required**

## Limitations

1. **Single-channel analysis**: Processes one signal at a time
2. **Sampling rate**: Optimized for 125-1000 Hz
3. **Signal quality**: Requires reasonably clean signals
4. **Beat detection**: May struggle with highly irregular rhythms

## References

1. Charlton, P.H. et al. (2019). "Modeling arterial pulse waves in healthy aging"
2. Pimentel, M.A.F. et al. (2012). "Incremental merge segmentation algorithm"
3. Orphanidou, C. et al. (2015). "Signal-quality indices for PPG"

---

**For detailed examples and tutorials, visit**: https://github.com/peterhcharlton/pulse-analyse/wiki/Examples