# Arterial Stiffness Prediction from Wrist Pulse Waveforms

## Project Overview

This project analyzes how **arterial stiffness** (as measured by carotid-femoral pulse wave velocity, cfPWV) affects the shape and features of **wrist (radial) artery pulse waveforms**. The goal is to use **machine learning** and signal processing to estimate central arterial stiffness non-invasively from peripheral (wrist) signals, potentially enabling accessible cardiovascular risk screening and wearable health applications.

## Motivation

Arterial stiffness is a powerful marker of vascular health. Measuring central stiffness directly (cfPWV) requires specialized equipment and expertise, but **peripheral pulse waves at the wrist are easy to record with consumer devices**. Understanding how stiffness alters the morphology of wrist waveforms could enable novel, practical health monitoring tools.

## Dataset

This project uses the **original HaeMod virtual subject pulse wave database**:

> Marie Willemet, Phil Chowienczyk and Jordi Alastruey.  
> "A database of virtual healthy subjects to assess the accuracy of foot-to-foot pulse wave velocities for estimation of aortic stiffness."  
> *American Journal of Physiology: Heart and Circulatory Physiology*, 309:H663-675, 2015.  
> [https://journals.physiology.org/doi/full/10.1152/ajpheart.00175.2015](https://journals.physiology.org/doi/full/10.1152/ajpheart.00175.2015)  
> [Dataset download and info](http://haemod.uk/original)

**Key facts:**
- 3,325 virtual healthy subjects, each with simulated arterial waveforms and physiological parameters.
- Multiple arterial sites, including the wrist (radial artery).
- Each subject includes blood pressure, flow, and area waveforms, plus computed indices (cfPWV, cardiac output, vessel geometry, etc.).
- Matlab-formatted data and search/extraction tools provided.

## Project Workflow

1. **Dataset Exploration**
   - Visualize example wrist (radial) pressure waveforms across the spectrum of stiffness (cfPWV).
   - Plot distribution of cfPWV and other parameters.
   - Examine how waveforms change with arterial stiffness.

2. **Feature Extraction**
   - Extract features from each wrist pulse waveform: pulse height, upstroke/downstroke timing and slope, area under the curve, systolic time, etc.
   - Correlate features with cfPWV and other indices.

3. **Data Export**
   - Combine features and cfPWV into a CSV file (`radial_features_labels.csv`) for machine learning.

4. **Machine Learning (Planned/Example)**
   - Train regression models (e.g., in Python) to predict cfPWV from waveform features.
   - Analyze feature importances and model accuracy.
   - (Optionally) Try end-to-end models using the entire waveform.

## Why This Matters

- **Scientific:** Links physiological modeling to modern data science, probing the connection between arterial stiffness and peripheral pulse morphology.
- **Clinical:** Success could enable at-home vascular health screening using wearable devices.
- **Technical:** Demonstrates a reproducible pipeline from simulated data → feature engineering → ML modeling.

## Usage

1. Download the HaeMod database from [http://haemod.uk/original](http://haemod.uk/original).
2. Run the Matlab scripts to extract and analyze features.
3. Use the exported CSV for machine learning (see `notebook.ipynb` for an example in Python).

## Acknowledgements

- Original dataset:  
  Willemet M, Chowienczyk P, Alastruey J. *Am J Physiol Heart Circ Physiol*, 2015.  
  [https://journals.physiology.org/doi/full/10.1152/ajpheart.00175.2015](https://journals.physiology.org/doi/full/10.1152/ajpheart.00175.2015)
- HaeMod Project: [http://haemod.uk/original](http://haemod.uk/original)

## Citation

If you use this project or data, please cite:

> Willemet M, Chowienczyk P, Alastruey J.  
> A database of virtual healthy subjects to assess the accuracy of foot-to-foot pulse wave velocities for estimation of aortic stiffness.  
> *Am J Physiol Heart Circ Physiol*. 2015 Sep 1;309(5):H663-75.  
> [https://doi.org/10.1152/ajpheart.00175.2015](https://doi.org/10.1152/ajpheart.00175.2015)

---

**Discussion and collaboration welcome!**  
If you're interested in signal processing, cardiovascular modeling, or ML, feel free to contribute or reach out.
