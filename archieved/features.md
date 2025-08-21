# 1. What are P, Q, A, and PPG in the Database?

| Symbol | Meaning                     | Units              | Physiological Interpretation                        | How Measured (in-vivo)         |
|--------|-----------------------------|--------------------|-----------------------------------------------------|--------------------------------|
| **P**  | Pressure                    | mmHg or Pascals    | Arterial blood pressure waveform                    | Tonometry, catheter, cuff      |
| **Q**  | Flow                        | m³/s               | Blood flow waveform in artery                       | Doppler ultrasound             |
| **A**  | Luminal Area                | m²                 | Cross-sectional area of artery                      | MRI, ultrasound (rare in vivo) |
| **PPG**| Photoplethysmogram (simulated) | a.u. (arbitrary) | Changes in blood volume (optical)                   | Optical PPG sensor             |

---

## Physiological Difference

### **P (Pressure)**
- **Directly reflects** the force of blood against artery walls.
- Contains all dynamic effects of heart ejection, reflection, and arterial compliance.
- **Closely linked** to arterial stiffness (stiffer arteries → higher pulse wave velocity → sharper upstroke, earlier return/reflection).

### **Q (Flow)**
- The **rate of blood** moving through the artery.
- Provides info on forward vs. reflected flow, but is **harder to measure non-invasively**.
- Can be influenced by stiffness but also by **cardiac output and peripheral resistance**.

### **A (Area)**
- The **size of the arterial lumen** during the cardiac cycle.
- Changes with pressure; **stiffer arteries expand less** for the same pressure.
- **Rarely measured directly** in clinics but highly informative in models.

### **PPG**
- **Optical signal** reflecting blood volume changes in the microvasculature (not a direct arterial signal).
- Influenced by pressure, but also **skin, tissue, and sensor factors**.
- Shape is related to the pressure wave, but further filtered by **tissue and optical path**.

---

## Which is Most Relevant for Predicting Arterial Stiffness?

### **Most Direct to Least Direct:**

1. **Pressure (P):**
   - Most direct and informative for arterial stiffness.
   - Central arterial pressure waveforms are the "gold standard" for cfPWV estimation.
   - Features like upstroke, augmentation, reflected wave arrival are classic stiffness markers.

2. **Area (A):**
   - Also directly reflects vessel stiffness (**less expansion = stiffer artery**).
   - But A is rarely available outside in silico or MRI/ultrasound.

3. **Flow (Q):**
   - Useful, especially when combined with P, but less often used alone for stiffness.

4. **PPG:**
   - Most practical (easily measured at wrist), but is an **indirect marker**.
   - PPG waveform shape is influenced by arterial stiffness, but also by peripheral vascular bed, tissue damping, and sensor properties.
   - Good PPG features (e.g., stiffness index, augmentation index, upstroke time) **can correlate with central stiffness**, but PPG-based predictions tend to be **noisier and less direct than pressure-based ones**.

---


# 2. Features: What and Why

## Features Table

| Feature          | Description                                                                                       | Medical Relevance to Arterial Stiffness and Pulse Morphology                                                                                      |
|------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Pulse Height** | Difference between peak (max) and trough (min) pressure in one cardiac cycle.                     | Stiffer arteries tend to transmit higher-pressure pulses more directly and may cause increased systolic (peak) pressures and pulse amplitude, especially in peripheral (wrist) locations. A high pulse height can signal higher pulse pressure, which is associated with vascular aging. |
| **Upstroke Time**| Time taken from minimum (end-diastolic) to maximum (systolic peak) pressure.                      | Shorter upstroke time is often seen in stiffer arteries: the pressure wave travels faster, so the upstroke is sharper. A rapid upstroke is a classical sign of decreased arterial compliance (i.e., increased stiffness). |
| **Upstroke Slope** | Maximum slope (steepest increase) of the pressure waveform during the upstroke.                 | Stiffer arteries produce sharper, steeper upstrokes. Clinically, the "steepness" of the upstroke (dP/dt_max) is sometimes used as a marker of arterial elasticity and ventricular contractility. |
| **Downstroke Slope** | Minimum slope (steepest decrease) after the systolic peak (i.e., rate of pressure fall during diastole). | Stiffer arteries may also alter the downstroke, often leading to a more rapid fall-off. Changes in the downstroke shape can indicate altered reflection and compliance in the vascular bed. |
| **Area Under Curve (AUC)** | The integral of the pressure waveform over the cardiac cycle (total "pressure-time" area). | The overall AUC can reflect total pressure exposure; in stiffer arteries, this may increase due to elevated systolic pressures and altered waveform shape. |
| **Systolic Time** | The fraction (or duration) of the cardiac cycle where pressure is above the mean value.          | Stiff arteries often alter the timing of the pulse waveform—especially by shortening systole and shifting the dicrotic notch. Systolic time shifts can indicate altered ventricular-arterial coupling and wave reflection. |

**Eel**: represents a scaling factor for the elastic modulus of the arteries in the simulation—a direct proxy for arterial wall stiffness. Higher Eel means stiffer arteries.

---

## Label: What and Why

| Label   | Description                                                                               | Medical Relevance                                                                                                          |
|---------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **cfPWV** | Carotid–femoral Pulse Wave Velocity: The speed at which the pressure wave travels between the carotid (neck) and femoral (groin) arteries. | This is the clinical "gold standard" for measuring central arterial stiffness. Higher cfPWV means stiffer arteries. By predicting cfPWV from peripheral wave features, you're assessing how well wrist morphology reflects true central stiffness. |

---

## Why These Features Matter for The Project

- **Wrist waveforms (radial artery):**  
  The wrist is a peripheral site, but its wave shape is influenced by the properties of the entire arterial tree. Stiffness "upstream" changes the shape, height, timing, and slopes of the pulse at the wrist.

- **All chosen features have a documented or plausible link to arterial stiffness**—they reflect either the speed/shape of the wave (upstroke time, slopes), its amplitude (pulse height), or timing (systolic time).

- **By relating these features to cfPWV,** we test if a machine learning model can "read" the wrist pulse and infer central artery health, which is a major goal in wearable health tech.

