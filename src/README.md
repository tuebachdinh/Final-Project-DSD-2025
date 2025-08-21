# Machine Learning Analysis of Wrist PPG for Arterial Stiffness Estimation Using the Charlton et al. In Silico Database

## Project Target

The goal of this project is to explore and quantify the relationship between arterial stiffness and wrist photoplethysmogram (PPG) waveform morphology, using machine learning methods. Specifically, we aim to:

- Understand how arterial stiffness (measured by pulse wave velocity, PWV) influences wrist PPG signals.
- Evaluate the feasibility of predicting central arterial stiffness directly from wrist PPG using supervised machine learning models.
- Compare the performance of ML-based predictions with classical feature-based (index-based) approaches.
- Provide insights that can facilitate non-invasive arterial health monitoring using wearable sensors.

## Dataset and Reference

This project utilizes the open in silico arterial pulse wave database published by Charlton et al. (2019):

**Reference:**
> Charlton PH, Mariscal Harana J, Vennin S, Li Y, Chowienczyk P, Alastruey J.  
> "Modeling arterial pulse waves in healthy aging: a database for in silico evaluation of hemodynamics and pulse wave indexes."  
> *American Journal of Physiology-Heart and Circulatory Physiology*, 317(5): H1062–H1085, 2019.  
> https://doi.org/10.1152/ajpheart.00218.2019

- The database is available at: https://github.com/peterhcharlton/pwdb
- The main file used: **pwdb_data.mat**
    - Contains simulated pulse waves (including wrist PPG), cardiovascular parameters, and hemodynamic features for thousands of "virtual subjects" aged 25–75.

## Project Plan

1. **Data Preparation**
    - Load the Charlton et al. dataset in MATLAB.
    - Filter for physiologically plausible virtual subjects.
    - Extract wrist (radial artery) PPG waveforms and corresponding central PWV values.

2. **Waveform Analysis**
    - Analyze and visualize how the shape of wrist PPG changes with arterial stiffness.
    - Group subjects by PWV tertiles and plot average PPG waveforms to illustrate morphological trends.

3. **Classical Feature-based Regression**
    - Extract classical PPG indexes (Reflection Index, Stiffness Index, Modified Aging Index).
    - Fit linear regression models to predict PWV from these indexes.
    - Benchmark performance (e.g., R-squared).

4. **Data-driven Machine Learning Modeling**
    - Train regression models (e.g., regression trees, other ML models) using the full wrist PPG waveform as input to predict PWV.
    - Evaluate model performance (correlation, error metrics) and compare to feature-based methods.

5. **Interpretation and Insight**
    - Interpret how waveform morphology encodes arterial stiffness, informed by both classical and ML models.
    - Discuss implications for non-invasive arterial health monitoring with wearable sensors.

6. **(Optional/Extensions)**
    - Investigate feature importance and interpretability of ML models.
    - Test the approach on real-world wrist PPG data if available.
    - Explore transfer learning, noise robustness, and domain adaptation.

## How to Use

- Requires MATLAB with the *Statistics and Machine Learning Toolbox*.
- Download `pwdb_data.mat` from the [Charlton et al. GitHub page](https://github.com/peterhcharlton/pwdb).
- See the main script for step-by-step data processing, analysis, and modeling.

## Acknowledgments

- Pulse wave database and in silico modeling:  
  *Peter H. Charlton, Jorge Mariscal Harana, Samuel Vennin, Ye Li, Phil Chowienczyk, Jordi Alastruey* (2019)

## Contact

For questions, please contact [Dinh Bach Tue / tue.dinh@aalto.fi].
