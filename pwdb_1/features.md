## Classic PPG-Derived Indices of Arterial Stiffness

Below are three widely used indices derived from the photoplethysmogram (PPG) waveform, their definitions, and clinical relevance.

---

### **Reflection Index (RI)**

- **Definition:**  
  The ratio of the amplitude of the second peak (or shoulder) to the amplitude of the first peak in the PPG waveform.
- **What it measures:**  
  Reflects the magnitude of the reflected pulse wave returning from the periphery, which increases with arterial stiffness.
- **Clinical recognition:**  
  RI is used in both clinical and research contexts as a surrogate marker for arterial stiffness.  
  [Reference: Millasseau et al., 2002](https://pubmed.ncbi.nlm.nih.gov/12160191/)  
  [Reference: Allen, 2007 - Review](https://pubmed.ncbi.nlm.nih.gov/17322588/)

---

### **Stiffness Index (SI)**

- **Definition:**  
  SI = Height (meters) / ΔT,  
  where ΔT is the time delay between the PPG's systolic peak and reflected wave peak (in the same pulse).
- **What it measures:**  
  Since pulse waves travel faster in stiffer arteries, a smaller ΔT (faster return) means a higher SI.
- **PPG-specific:**  
  ΔT is measured directly from the PPG waveform.
- **Clinical recognition:**  
  SI has been validated against gold-standard PWV and correlates with age and vascular disease risk.  
  [Reference: Millasseau et al., 2002](https://pubmed.ncbi.nlm.nih.gov/12160191/)  
  [Reference: Takazawa et al., 1998](https://pubmed.ncbi.nlm.nih.gov/9662347/)

---

### **Modified Aging Index (AGImod)**

- **Definition:**  
  A more recent feature that combines information from the PPG's upstroke and downstroke to reflect vascular aging, designed to improve robustness over traditional indexes.
- **What it measures:**  
  Captures complex changes in waveform shape that occur with arterial aging/stiffening.
- **Clinical recognition:**  
  AGImod has been proposed as a robust index for vascular aging, validated in both simulation and real human data.  
  [Reference: Yoon et al., 2018](https://www.mdpi.com/1424-8220/18/7/2103)  
  [Reference: Ding et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34941373/)

---

### **Why use them?**
- **Simple and interpretable:** Each has a clear physiological meaning and is easy to compute from PPG.
- **Clinically relevant:** Shown in simulation and real human data to correlate with central arterial stiffness and cardiovascular risk ([Allen, 2007](https://pubmed.ncbi.nlm.nih.gov/17322588/); [Millasseau, 2002](https://pubmed.ncbi.nlm.nih.gov/12160191/)).
- **Baseline for ML:** Provide a meaningful baseline for comparison against more complex machine learning methods.

---

### **Summary Table**

| Index   | Input           | What it measures                    | PPG-specific? | Interpretation                        |
|---------|-----------------|-------------------------------------|---------------|----------------------------------------|
| **RI**      | PPG waveform      | Pulse wave reflection amplitude    | Yes           | ↑ RI = ↑ reflections = stiffer         |
| **SI**      | PPG waveform + height | Pulse wave velocity              | Yes           | ↑ SI = faster PWV = stiffer            |
| **AGImod**  | PPG waveform      | Combined/shape-based aging index  | Yes           | ↑ AGImod = advanced vascular age        |

---

#### **References**

- [Millasseau SC, et al. "Contour analysis of the photoplethysmographic pulse measured at the finger." Br J Clin Pharmacol. 2002.](https://pubmed.ncbi.nlm.nih.gov/12160191/)
- [Takazawa K, et al. "Assessment of Vasoactive Agents and Vascular Aging by the Second Derivative of Photoplethysmogram Waveform." Hypertension. 1998.](https://pubmed.ncbi.nlm.nih.gov/9662347/)
- [Allen J. "Photoplethysmography and its application in clinical physiological measurement." Physiol Meas. 2007. (Review)](https://pubmed.ncbi.nlm.nih.gov/17322588/)
- [Yoon H, et al. "Photoplethysmogram Analysis for Aging Index Development: A Simulation Study." Sensors. 2018.](https://www.mdpi.com/1424-8220/18/7/2103)
- [Ding Z, et al. "Machine learning-based assessment of vascular aging using PPG waveform features." Biomed Eng Online. 2021.](https://pubmed.ncbi.nlm.nih.gov/34941373/)

---
