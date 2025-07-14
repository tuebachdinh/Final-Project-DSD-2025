# pwdb_data.mat

*all the important data in a single Matlab ® file*

This page provides details of the contents of the *pwdb_data.mat* file. The file contains a single structure called *data*, with three fields:

* *[data.config](#dataconfig)*: Details of the model configuration used for each virtual subject
* *[data.waves](#datawaves)*: The simulated PWs at common measurement sites for each virtual subject
* *[data.haemods](#datahaemods)*: Haemodynamic parameters extracted from PWs for each virtual subject
* *[data.pw_inds](#datapw_inds)*: PW indices extracted from PWs for each virtual subject
* *[data.plausibility](#dataplausibility)*: The haemodynamic plausibility of each virtual subject

The contents of each of these fields are now described.

## *data.config*

This field contains details of the model configuration used for each virtual subject. It contains the following variables:

> Note that each of these are the values prescribed to the model before running the simulations. Therefore, the values of haemodynamic variables (such as heart rate) will be close to the actual simulated values, but not exactly the same.

Variable | Contents
--- | ---
*baseline_sim_for_all* | A logical indicating which of the virtual subjects is the baseline subject (*i.e.* the 25-year old subject with all of its cardiovascular properties set to baseline values).
*baseline_sim_for_age* | A logical indicating which virtual subjects are the baseline subjects for each age group (*i.e.* the subjects with all of their cardiovascular properties set to the baseline value for their age).
*hr* | The heart rate (HR) prescribed for each virtual subject, measured in beats per minute (bpm).
*hr_SD* | The number of standard deviations of the heart rate (HR) away from the age-specific baseline value, for each virtual subject.
*sv* | The stroke volume (SV) prescribed for each virtual subject, measured in millilitres (ml).
*sv_SD* | The number of standard deviations of the stroke volume (SV) away from the age-specific baseline value, for each virtual subject.
*lvet* | The left ventricular ejection time (LVET) prescribed for each virtual subject, measured in milliseconds (ms).
*lvet_SD* | The number of standard deviations of the left ventricular ejection time (LVET) away from the age-specific baseline value, for each virtual subject.
*pft* | The aortic peak flow time (PFT) prescribed for each virtual subject, measured in milliseconds (ms).
*pft_SD* | The number of standard deviations of the aortic peak flow time (PFT) away from the age-specific baseline value, for each virtual subject.
*rfv* | The reverse flow volume (RFV) prescribed for each virtual subject, measured in millilitres (ml).
*rfv_SD* | The number of standard deviations of the reverse flow volume (RFV) away from the age-specific baseline value, for each virtual subject.
*dbp* | The diastolic blood pressure (DBP) prescribed for each virtual subject, measured in Pascals (Pa).
*dbp_SD* | The number of standard deviations of the diastolic blood pressure (DBP) away from the age-specific baseline value, for each virtual subject.
*mbp* | The mean blood pressure (MBP) prescribed for each virtual subject, measured in mmHg.
*mbp_SD* | The number of standard deviations of the mean blood pressure (MBP) away from the age-specific baseline value, for each virtual subject.
*pvr* | The peripheral vascular resistance (PVR) prescribed for each virtual subject, measured in *tbc*.
*pvr_SD* | The number of standard deviations of the peripheral vascular resistance (PVR) away from the age-specific baseline value, for each virtual subject.
*pvc* | The scaling factor applied to the baseline 25-year old's peripheral vascular compliance (PVC) prescribed for each virtual subject.
*pvc_SD* | The number of standard deviations of the PVC scaling factor away from the age-specific baseline value, for each virtual subject.
*p_out* | The outflow blood pressure (Pout) prescribed for each virtual subject, measured in Pascals (Pa).
*p_out_SD* | The number of standard deviations of the outflow blood pressure (Pout) away from the age-specific baseline value, for each virtual subject.
*len* | The scaling factor applied to the age-specific baseline subject's proximal aortic length prescribed for each virtual subject.
*len_SD* | The number of standard deviations of the proximal aortic length away from the age-specific baseline value, for each virtual subject.
*dia* | The scaling factor applied to the age-specific baseline subject's larger artery diameter prescribed for each virtual subject.
*dia_SD* | The number of standard deviations of the larger artery diameter away from the age-specific baseline value, for each virtual subject.
*dia* | The scaling factor applied to the age-specific baseline subject's pulse wave velocity (PWV) prescribed for each virtual subject.
*dia_SD* | The number of standard deviations of the pulse wave velocity (PWV) away from the age-specific baseline value, for each virtual subject.
*age* | The age of each virtual subject (in years).

*data.config* also contains several structures. The contents of these structures are listed below:

### *data.config.variations*

*variations* fields | Contents
--- | ---
*param_names* | contains the names of the model input parameters
*params* | contains the number of standard deviations of each parameter away from the age-specific baseline value, for each virtual subject.

### *data.config.desired_chars*

*desired_chars* fields | Contents
--- | ---
*pwv_cf* | The desired carotid-femoral PWV (m/s) values for each subject
*pwv_fa* | The desired femoral-ankle PWV (m/s) values for each subject
*pwv_br* | The desired brachial-radial PWV (m/s) values for each subject
*dia_asc_a* | The desired diameter of the ascending aorta (mm) for each subject
*dia_desc_thor_a* | The desired diameter of the descending thoracic aorta (mm) for each subject
*dia_abd_a* | The desired diameter of the abdominal aorta (mm) for each subject
*dia_car* | The desired diameter of the carotid artery (mm) for each subject
*len_prox_a* | The desired length of the proximal aorta (mm) for each subject

### *data.config.network*

*network* fields | Contents
--- | ---
*segment_name* | The names of each of the 116 arterial segments used in the model
*inlet_node* | The inlet nodes of each of the 116 arterial segments, for each virtual subject
*outlet_node* | The outlet nodes of each of the 116 arterial segments, for each virtual subject
*length* | The lengths of each of the 116 arterial segments, for each virtual subject (in metres)
*inlet_radius* | The inlet radii of each of the 116 arterial segments, for each virtual subject (in metres)
*outlet_radius* | The outlet radii of each of the 116 arterial segments, for each virtual subject (in metres)

### *data.config.constants*

*constants* fields | Contents
--- | ---
*k* | The values of k1, k2 and k3 used in the equation describing arterial stiffness (*Eh*), for each virtual subject
*rho* | Blood density (kg/m3) for each virtual subject
*gamma_b0* | The value of *b0* used in the equation describing arterial wall viscosity, for each virtual subject
*gamma_b1* | The value of *b1* used in the equation describing arterial wall viscosity, for each virtual subject
*mu* | Blood viscosity (Pa s) for each virtual subject
*alpha* | The viscosity profile coefficient (unitless) for each virtual subject
*p_drop* | The assumed drop in mean arterial blood pressure from the aortic root to the periphery for each virtual subject

### *data.config.system_chars*

*system_chars* fields | Contents
--- | ---
*pvr* | The values of the peripheral vascular resistance (PVR), measured in Pa s / m3, for each virtual subject
*pvc* | The values of the total compliance, measured in m3 / Pa, for each virtual subject
*pvc_iw* | The values of the impedance-weighted peripheral compliance, measured in m3 / Pa, for each virtual subject
*ac* | The values of the total arterial compliance, measured in m3 / Pa, for each virtual subject
*c* | The values of the total compliance, measured in m3 / Pa, for each virtual subject
*tau* | The time constant, measured in s, for each virtual subject

## *data.waves*

The *data.waves* structure contains the following fields.

Field | Contents
--- | ---
*fs* | Wave sampling frequency, measured in Hz
*P_#####* | Arterial blood pressure wave (P), measured in mmHg, for each virtual subject
*U_#####* | Arterial blood flow velocity wave (U), measured in m/s, for each virtual subject
*A_#####* | Arterial luminal area wave (A), measured in m3, for each virtual subject
*PPG_#####* | Photoplethysmogram (PPG) wave for each virtual subject (arbitrary units)

In each field name, *#####* denotes an arterial site (such as *AorticRoot*). The locations of these sites are as follows:

<a name="arterial_sites" />

Site | Location
--- | ---
*AorticRoot* | The aortic root, *i.e.* at the start of segment no. 1.
*ThorAorta* | In the thoracic aorta, *i.e.* at the end of segment 18.
*AbdAorta* | In the abdominal aorta, *i.e.* at the start of segment 39.
*IliacBif* | At the iliac bifurcation (on the aortic side rather than in an iliac artery), *i.e.* at the end of segment 41.
*Carotid* | In the left carotid artery, *i.e.* half way along segment 15.
*SupTemporal* | At the end of the left superior temporal artery, *i.e.* at the end of segment 87. This location was used to approximate pulse waves at the ear.
*SupMidCerebral* | At the end of the Left Superior Middle Cerebral artery, *i.e.* at the end of segment 72.
*Brachial* | In the left brachial artery, at approximately the location where a brachial cuff measurement would be taken, *i.e.* three quarters of the way along the brachial artery (*i.e.* in the upper arm, segment 21).
*Radial* | At the end of the left radial artery (*i.e.* at the wrist, segment 22).
*Digital* | At the end of one of the left digital arteries (*i.e.* at the finger, segment 112).
*CommonIliac* | Half way along the left common iliac artery (segment 44).
*Femoral* | Half way along the left femoral artery (segment 46).
*AntTibial* | At the end of the left anterior tibial artery (*i.e.* at the ankle, segment 49).

The *data.waves* structure also contains a structure called *onset_times*. This structure contains the pulse onset times for each pulse wave (measured in secs). This is helpful for looking at pulse wave propagation, including calculating pulse wave velocities.

## *data.haemods*

This structure contains the following fields, with a value for each virtual subject. Note that some fields are followed by an abbreviation indicating arterial site:
* *a* - aortic root
* *b* - brachial artery
* *f* - finger

Field | Contents
---|---
*age* | age, in years
*HR* | heart rate, bpm
*SV* | stroke volume, ml
*CO* | cardiac output, l/min
*LVET* | left ventricular ejection time (*i.e.* duration of systole), ms
*dPdt* | maximum value of the time-derivative of the pressure wave, mmHg/s
*PFT* | time of peak aortic flow, ms
*RFV* | reverse flow volume, ml
*SBP* | systolic blood pressure, mmHg
*DBP* | diastolic blood pressure, mmHg
*MBP* | mean blood pressure, mmHg
*PP* | pulse pressure, mmHg
*PP_amp* | pulse pressure amplification (*i.e* brachial PP divided by aortic root PP), ratio
*MBP_drop_finger* | pressure drop from aortic root to digital artery, mmHg
*MBP_drop_ankle* | pressure drop from aortic root to anterior tibial artery, mmHg
*SBP_diff* | brachial SBP minus aortic root SBP, mmHg
*SMBP* | systolic minus mean blood pressure, mmHg
*P1* | blood pressure at the time of P1, mmHg
*P1t* | time of P1, secs
*P2* | blood pressure at the time of P2, mmHg
*P2t* | time of P2, secs
*Ps* | blood pressure at systolic peak, mmHg
*Pst* | time of systolic peak, secs
*AI* | augmentation index, calculated from: *(P2-P1)/PP*
*AP* | augmentation index, calculated from: *(P2-P1)*
*Tr* | time to reflected wave, ms, equal to P1t_c
*IAD* | inter-arm difference in SBP, mmHg, calculated from: *SBP(right brachial) - SBP(left brachial)*
*IADabs* | absolute inter-arm difference in SBP, mmHg
*PWV_a* | pulse wave velocity (m/s) measured between the aortic root and iliac bifurcation (foot-to-foot between pressure waves)
*PWV_cf* | pulse wave velocity (m/s) measured between the carotid and femoral arteries (foot-to-foot between pressure waves)
*PWV_cr* | pulse wave velocity (m/s) measured between the carotid and radial arteries (foot-to-foot between pressure waves)
*PWV_ca* | pulse wave velocity (m/s) measured between the carotid artery and the ankle (foot-to-foot between pressure waves)
*PWV_cb* | pulse wave velocity (m/s) measured between the carotid and the brachial arteries (foot-to-foot between pressure waves)
*PWV_bf* | pulse wave velocity (m/s) measured between the brachial and femoral arteries (foot-to-foot between pressure waves)
*PWV_br* | pulse wave velocity (m/s) measured between the brachial and radial arteries (foot-to-foot between pressure waves)
*PWV_ba* | pulse wave velocity (m/s) measured between the brachial artery and the ankle (foot-to-foot between pressure waves)
*PWV_rf* | pulse wave velocity (m/s) measured between the radial and femoral arteries (foot-to-foot between pressure waves)
*PWV_ra* | pulse wave velocity (m/s) measured between the radial artery and the ankle (foot-to-foot between pressure waves)
*PWV_fa* | pulse wave velocity (m/s) measured between the femoral and anterior tibial (ankle) arteries (foot-to-foot between pressure waves)
*PWV_a* | pulse wave velocity (m/s) measured between the aortic root and iliac bifurcation (theoretical)
*PWV_cf* | pulse wave velocity (m/s) measured between the carotid and femoral arteries (theoretical)
*PWV_cr* | pulse wave velocity (m/s) measured between the carotid and radial arteries (theoretical)
*PWV_ca* | pulse wave velocity (m/s) measured between the carotid artery and the ankle (theoretical)
*PWV_cb* | pulse wave velocity (m/s) measured between the carotid and the brachial arteries (theoretical)
*PWV_bf* | pulse wave velocity (m/s) measured between the brachial and femoral arteries (theoretical)
*PWV_br* | pulse wave velocity (m/s) measured between the brachial and radial arteries (theoretical)
*PWV_ba* | pulse wave velocity (m/s) measured between the brachial artery and the ankle (theoretical)
*PWV_rf* | pulse wave velocity (m/s) measured between the radial and femoral arteries (theoretical)
*PWV_ra* | pulse wave velocity (m/s) measured between the radial artery and the ankle (theoretical)
*PWV_fa* | pulse wave velocity (m/s) measured between the femoral and anterior tibial (ankle) arteries (theoretical)
*svr* | systemic vascular resistance (10^6 Pa s / m3)
*dia_asc_a* | ascending aorta diameter (mm)
*dia_desc_thor_a* | descending thoracic aorta diameter (mm)
*dia_abd_a* | abdominal aorta diameter (mm)
*dia_car* | carotid diameter (mm)
*len_prox_a* | length of proximal aorta (mm)
*RI* | reflection index, measured from the left digital PPG
*SI* | stiffness index (m/s), measured from the left digital PPG, assuming a height of 1.75m
*AGI_mod* | modified ageing index, measured from the left digital PPG
*pvr* | peripheral vascular resistance (Pa s / m3)
*pvc* | total compliance (m3 / Pa)
*pvc_iw* | impedance-weighted peripheral compliance (m3 / Pa)
*ac* | total arterial compliance (m3 / Pa)
*c* | total compliance (m3 / Pa)
*tau* | time constant (s)

## *data.pw_inds*

This field contains pulse wave indices measured from simulated pulse waves for each virtual subject.  
Indices are provided in the format and with the definitions described below.  
The data is organized identically to the `pwdb_pw_indices.csv` spreadsheet.

### Column Format

Most column headers consist of three parts, separated by underscores, e.g., `AorticRoot_SBP_V`:
- **Measurement site** (e.g., `AorticRoot`), as defined in the [arterial sites](#arterial_sites) section.
- **Variable type** (e.g., `SBP`), as defined below.
- **Value or time**:
    - `V`: value (actual parameter value)
    - `T`: time (in seconds)
    - If no suffix is provided, the column refers to the value.

### Variable Types

| Column Header | Details |
|---|---|
| Age | Age, in years |
| SBP | Systolic blood pressure, in mmHg |
| DBP | Diastolic blood pressure, in mmHg |
| MBP | Mean blood pressure, in mmHg |
| PP | Pulse pressure, in mmHg |
| Qmax | Maximum flow rate, in m³/s |
| Qmin | Minimum flow rate, in m³/s |
| Qmean | Mean flow rate, in m³/s |
| Qtotal | Total flow during one cardiac cycle, in m³ |
| Umax | Maximum flow velocity, in m/s |
| Umin | Minimum flow velocity, in m/s |
| Umean | Mean flow velocity, in m/s |
| Amax | Maximum luminal area, in m² |
| Amin | Minimum luminal area, in m² |
| Amean | Mean luminal area, in m² |
| P1in | p1, measured at the inflection point, in mmHg |
| P1pk | p1, measured at the peak, in mmHg |
| P2in | p2, measured at the inflection point, in mmHg |
| P2pk | p2, measured at the peak, in mmHg |
| Psys | Systolic pressure peak, in mmHg |
| Pms | First derivative of pressure at time of maximum slope (max dP/dt), in mmHg/s |
| AI | Augmentation index, calculated from P1in and P2pk |
| AP | Augmentation pressure, calculated from P1in and P2pk |
| PTT | Time between the onset of the pressure wave at the aortic root and the onset of the current pressure wave, in seconds |
| PPGa | 'a' point on the second derivative of the PPG (au/s²). The PPG is measured in arbitrary units, scaled 0–1. |
| PPGb | 'b' point on the second derivative of the PPG (au/s²) |
| PPGc | 'c' point on the second derivative of the PPG (au/s²) |
| PPGd | 'd' point on the second derivative of the PPG (au/s²) |
| PPGe | 'e' point on the second derivative of the PPG (au/s²) |
| PPGsys | Systolic peak on the PPG, in au |
| PPGdia | Diastolic peak on the PPG, in au |
| PPGdic | Dicrotic notch on the PPG, in au |
| PPGms | First derivative of the PPG at time of maximum slope (max dPPG/dt), in au/s |
| RI | Reflection index, calculated from the PPG wave |
| SI | Stiffness index, calculated from the PPG wave (assuming a height of 1.75 m) |
| AGI_mod | Modified ageing index, calculated from the PPG wave |

**Note:** PPG-derived values are only provided at peripheral sites.

---

**For details of the measurement sites, see the section above.**  


## *data.plausibility*

This structure reports the physiological plausibility of each virtual subject's BPs. Plausibility was assessed using the approach described in the [*Generating a Database of Arterial Pulse Waves* Section](https://journals.physiology.org/doi/full/10.1152/ajpheart.00218.2019#_i11) of [this article](https://peterhcharlton.github.io/pwdb/pwdb_article.html).

It contains the following fields, with a value for each virtual subject. Note that some fields are followed by an abbreviation indicating arterial site:
* *implaus_sims* - the subject numbers (_..._el_) and the cardiovascular variations used in their simulations (_..._vars_, as defined by _var_names_) which were deemed to be implausible for each plausibility criterion (brachial SBP - _SBP_b_; brachial DBP - _DBP_b_; brachial PP - _PP_b_; brachial MBP - _MBP_b_; aortic SBP - _SBP_a_; aortic PP - _PP_a_; PP amplification - _PP_amp_; these are described in [the accompanying article](https://peterhcharlton.github.io/pwdb/pwdb_article.html)).
* *all_implaus_sims* - similar information on all of the physiologically implausible virtual subjects (rather than separated according to each plausibility criterion).
* *plausibility_log* - a logical vector indicating whether or not each virtual subject was deemed to be physiologically plausible: 1- physiologically plausible; 0 - physiologically implausible.