# 🌊 TARAL — OceanEmbed
## Thermal Analysis and Reconstruction using AI for Ocean Layers

**Smart India Hackathon 2026 | Problem Statement ID: 26066**

> **Problem Statement:** OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.

| Project Detail | Information |
|---|---|
| Project Name | TARAL — OceanEmbed |
| Full Form | Thermal Analysis and Reconstruction using AI for Ocean Layers |
| Problem Statement ID | SIH26066 |
| Category | Software |
| Theme | Disaster Management |
| Team | B2S — Bharat Through Space |
| Study Region | North Indian Ocean |
| Spatial Resolution | 0.25° × 0.25° |
| Target Depth Range | Surface to 1,000 m |
| Target Depth Levels | 15 standard depths |

### Team Members

1. Lalit Hire
2. Amit Joshi
3. Saad Shaikh
4. Lochana Pawar
5. Rushikesh Bhavar
6. Akshata Lonari

---

# Table of Contents

- [1. Project Information](#1-project-information)
  - [1.1 Problem Statement](#11-problem-statement)
  - [1.2 Background](#12-background)
  - [1.3 Why We Chose This Problem](#13-why-we-chose-this-problem)
  - [1.4 Problem Definition](#14-problem-definition)
  - [1.5 Study Region](#15-study-region)
  - [1.6 Objectives](#16-objectives)
  - [1.7 Applications](#17-applications)
  - [1.8 Expected Outcomes](#18-expected-outcomes)
- [2. Proposed Solution](#2-proposed-solution)
  - [2.1 Overview](#21-overview-of-the-proposed-solution)
  - [2.2 Data Collection and Integration](#22-data-collection-and-integration)
  - [2.3 Preprocessing and Harmonization](#23-data-preprocessing-and-harmonization)
  - [2.4 Satellite Embedding Engine](#24-satellite-embedding-engine)
  - [2.5 Deep Learning Reconstruction](#25-deep-learning-reconstruction)
  - [2.6 Validation and Evaluation](#26-validation-and-evaluation)
  - [2.7 Overall Workflow](#27-overall-workflow)
- [3. Datasets](#3-datasets)
  - [3.1 OISST](#31-oisst-dataset)
  - [3.2 GlobCurrent](#32-globcurrent-dataset)
  - [3.3 Argo](#33-argo-dataset)
  - [3.4 CCMP](#34-ccmp-dataset)
  - [3.5 GLORYS](#35-glorys-dataset)
  - [3.6 OSCAR](#36-oscar-dataset)
  - [3.7 OISSS](#37-oisss-dataset)
  - [3.8 DUACS](#38-duacs-dataset)
- [4. Validation and Reproducibility](#4-validation-and-reproducibility)
- [5. Limitations and Important Considerations](#5-limitations-and-important-considerations)

---

# 1. Project Information

## 1.1 Problem Statement

The problem statement, identified as **Problem Statement ID 26066** under Smart India Hackathon (SIH) 2026, is titled:

**OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.**

The objective is to develop a satellite embedding-based deep learning framework for reconstructing depth-wise subsurface ocean temperature from daily surface observations at a spatial resolution of 0.25°.

The study focuses on the North Indian Ocean, covering:

- **Latitude:** 5°N to 30°N
- **Longitude:** 45°E to 105°E
- **Spatial resolution:** 0.25° × 0.25°
- **Target depth:** Surface to 1,000 m
- **Target levels:** 15 standard depth levels

Subsurface temperature is important for understanding ocean circulation, heat content, stratification, climate variability, air–sea interactions, and marine ecosystems. However, direct measurements are spatially and temporally limited and primarily rely on in-situ systems such as Argo floats, moored buoys, gliders, and ships.

Satellite observations provide large-scale information about surface ocean conditions. Variables such as Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Level Anomaly (SLA), surface currents, and winds provide indirect information about subsurface processes.

The proposed framework aims to learn relationships between these surface observations and subsurface temperature using satellite embeddings and deep learning.

## 1.2 Background

Subsurface ocean temperature is a fundamental variable for understanding ocean circulation, upper-ocean heat content, stratification, climate variability, air–sea interactions, and marine ecosystems.

Accurate vertical temperature profiles are important for applications such as:

- Marine heatwave monitoring
- Fisheries and marine ecosystem assessment
- Ocean data assimilation
- Ocean circulation studies
- Climate variability analysis
- Air–sea interaction research

Direct subsurface measurements are spatially and temporally limited. They mainly rely on Argo profiling floats, moored buoys, gliders, and ship-based observations, which cannot provide continuous basin-scale temperature fields everywhere.

Satellite observations offer continuous, large-scale information about surface ocean conditions. SST, SSS, SLA, surface currents, and winds contain indirect signatures of subsurface processes through mechanisms such as thermocline displacement, mesoscale eddies, vertical mixing, ocean transport, and ocean–atmosphere coupling.

Advances in Artificial Intelligence (AI), Deep Learning (DL), and representation learning make it possible to learn complex relationships from multidimensional ocean data.

Satellite embeddings can transform surface observations into compact numerical representations that capture useful spatial patterns and nonlinear relationships between surface conditions and subsurface temperature structure.

## 1.3 Why We Chose This Problem

We selected OceanEmbed because subsurface ocean temperature is important for understanding ocean dynamics, climate variability, air–sea interactions, and marine ecosystems, while continuous large-scale subsurface observations remain limited.

The problem offers an opportunity to address this limitation using satellite observations such as SST, SSS, SLA, surface currents, and winds.

The availability of complementary ocean datasets also enables the development of a unified data-driven framework using AI and deep learning.

The project has potential applications in marine heatwave monitoring, fisheries, data assimilation, ocean circulation, and climate studies.

OceanEmbed therefore combines an important oceanographic challenge with modern deep learning techniques to develop a framework for estimating subsurface ocean temperature from surface observations.

## 1.4 Problem Definition

OceanEmbed aims to reconstruct the vertical structure of subsurface ocean temperature using daily surface ocean observations.

### Input variables

- Sea Surface Temperature (SST)
- Sea Surface Salinity (SSS)
- Sea Surface Height (SSH) or Sea Level Anomaly (SLA)
- Eastward and northward surface current components
- Eastward and northward surface wind components

### Target output

The target is ocean temperature at the following 15 standard depth levels:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

The problem is formulated as a supervised learning task in which multidimensional surface ocean observations are used to estimate corresponding depth-wise subsurface temperature fields.

## 1.5 Study Region

| Parameter | Definition |
|---|---|
| Region | North Indian Ocean |
| Latitude | 5°N to 30°N |
| Longitude | 45°E to 105°E |
| Spatial resolution | 0.25° × 0.25° |
| Input cadence | Daily surface observations |
| Vertical output | 15 standard depth levels |
| Maximum target depth | 1,000 m |
| Major areas | Arabian Sea, Bay of Bengal, surrounding North Indian Ocean waters |

## 1.6 Objectives

The main objective is to develop a deep learning-based framework for reconstructing subsurface ocean temperature from surface satellite observations.

Specific objectives:

1. Collect and integrate relevant surface and subsurface ocean observation datasets.
2. Preprocess and harmonize datasets across spatial, temporal, and variable dimensions.
3. Develop a satellite embedding-based approach for extracting meaningful representations from multidimensional surface observations.
4. Develop a deep learning framework that learns relationships between surface observations and subsurface temperature structure.
5. Reconstruct subsurface temperature at 15 standard depths from the surface to 1,000 m.
6. Generate daily reconstructed temperature fields at 0.25° × 0.25° resolution for the North Indian Ocean.
7. Evaluate reconstructed fields using suitable metrics and independent Argo observations.
8. Demonstrate the potential of the framework for marine heatwave monitoring, fisheries, data assimilation, ocean circulation, and climate studies.

## 1.7 Applications

### Marine Heatwave Monitoring

Understand the vertical extent and evolution of marine heatwaves, including temperature anomalies below the ocean surface.

### Fisheries

Assess ocean conditions that influence marine ecosystems, fish habitats, and fisheries.

### Data Assimilation

Provide additional temperature estimates for ocean data assimilation systems, subject to suitable validation.

### Ocean Circulation Studies

Study temperature distributions, circulation patterns, and related physical processes.

### Climate Variability

Analyze subsurface ocean changes associated with climate variability.

### Air–Sea Interaction

Support research into interactions between the ocean and atmosphere.

### Marine Ecosystems

Improve understanding of environmental conditions affecting marine ecosystems.

## 1.8 Expected Outcomes

1. A harmonized dataset containing relevant surface and subsurface ocean observations for the North Indian Ocean.
2. A standardized preprocessing pipeline to handle differences in spatial and temporal resolution, formats, and dataset characteristics.
3. A satellite embedding component for generating numerical representations from multidimensional surface observations.
4. A deep learning framework for reconstructing subsurface ocean temperature.
5. Daily subsurface temperature estimates at 0.25° × 0.25° spatial resolution.
6. Reconstructed temperature profiles at 15 standard depths from the surface to 1,000 m.
7. Evaluation using suitable statistical metrics and independent Argo observations.
8. A proof-of-concept implementation demonstrating the feasibility of subsurface temperature reconstruction from surface observations.

---

# 2. Proposed Solution

## 2.1 Overview of the Proposed Solution

The proposed OceanEmbed framework aims to reconstruct subsurface ocean temperature from surface ocean observations using a satellite embedding-based deep learning approach.

Multiple ocean datasets are collected, processed, and harmonized to ensure compatibility across spatial, temporal, and variable characteristics.

Surface variables, including SST, SSS, SLA, surface currents, and surface winds, are used as model inputs.

These observations are transformed into numerical representations through a satellite embedding component. The resulting embeddings are then supplied to a deep learning reconstruction model that learns relationships between surface ocean conditions and subsurface temperature structure.

The framework is designed to produce depth-wise subsurface temperature fields for the North Indian Ocean at 0.25° × 0.25° spatial resolution, covering 15 standard depths from the surface to 1,000 m.

The reconstructed fields are evaluated using suitable metrics and independent Argo observations.

## 2.2 Data Collection and Integration

The selected datasets provide complementary information about ocean surface conditions and subsurface temperature.

| Dataset | Main variable or information | Intended role |
|---|---|---|
| OISST | Sea Surface Temperature | Surface input |
| GlobCurrent | Surface current components | Candidate current input |
| Argo | In-situ temperature and salinity profiles | Independent validation |
| CCMP | 10 m surface winds | Surface input |
| GLORYS | Gridded ocean physics and subsurface fields | Candidate training target or reference |
| OSCAR | Surface currents | Candidate current input or alternative |
| OISSS | Sea Surface Salinity | Surface input |
| DUACS | Sea surface height, SLA, and derived currents | Surface input |

These datasets differ in variables, spatial and temporal resolutions, coverage periods, and file formats. Their characteristics must be examined before integration.

The integration process should align spatial coordinates, time representations, units, variable names, and missing-data conventions.

Argo profiles are intended to provide in-situ reference observations for validation. If GLORYS or another gridded product is used as the training target, that role must be explicitly documented and kept distinct from independent validation.

## 2.3 Data Preprocessing and Harmonization

The datasets come from different sources and may differ in spatial grids, temporal resolutions, coordinate conventions, units, file formats, and coverage.

The planned preprocessing steps are:

1. **Data inspection:** Examine variables, dimensions, coordinates, temporal coverage, and metadata.
2. **Quality checking:** Identify missing values, invalid values, duplicates, and other data-quality issues.
3. **Spatial harmonization:** Transform eligible datasets to a common 0.25° × 0.25° grid for the study region.
4. **Temporal harmonization:** Align observations to a daily time axis while accounting for different sampling cadences.
5. **Variable standardization:** Ensure consistent units, names, and representations.
6. **Missing-data handling:** Identify and appropriately handle missing observations.
7. **Data alignment:** Match spatial and temporal observations across datasets.
8. **Dataset splitting:** Separate training, validation, and test data to prevent leakage.
9. **Provenance tracking:** Record data sources, versions, preprocessing decisions, and quality-control rules.

The final preprocessing procedure must be determined through dataset auditing. Dataset-specific transformations and decisions should be documented for reproducibility.

## 2.4 Satellite Embedding Engine

The satellite embedding engine transforms multidimensional surface ocean observations into numerical representations called **embeddings**.

The input may include:

- SST
- SSS
- SSH/SLA
- Surface current components
- Surface wind components
- Spatial and temporal features

Instead of relying only on raw variables, the embedding stage is intended to learn compact representations of important spatial and physical patterns.

These representations are supplied to the reconstruction model to help it learn relationships between surface ocean conditions and subsurface temperature structure.

The exact embedding architecture and feature extraction method should be selected through experimentation and evaluation. The implementation should document whether embeddings are learned end-to-end, precomputed, or generated by a separate encoder.

## 2.5 Deep Learning Reconstruction

The reconstruction component estimates subsurface ocean temperature from surface observations and their learned representations.

The model is intended to learn spatial patterns and nonlinear relationships within ocean observations and generate temperature estimates at 15 predefined depth levels:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

Potential model architectures include:

- Convolutional Neural Networks (CNNs)
- Vision Transformers
- Autoencoders
- Graph Neural Networks (GNNs)
- Attention-based hybrid architectures
- CNN–LSTM or other spatiotemporal models

The final architecture should be selected based on suitability for the reconstruction task, computational requirements, and validation performance.

The intended output is a depth-wise subsurface ocean temperature field for the North Indian Ocean.

## 2.6 Validation and Evaluation

The reconstructed fields must be evaluated to assess accuracy and reliability.

Independent subsurface observations, particularly quality-controlled Argo profiles, should be used for validation. Reconstructed temperatures should be compared with observations at matching locations, dates, and depth levels wherever suitable observations are available.

### Recommended evaluation metrics

| Metric | Purpose |
|---|---|
| Root Mean Square Error (RMSE) | Measures prediction error and penalizes large errors |
| Mean Absolute Error (MAE) | Measures average absolute prediction error |
| Bias | Measures the average signed difference between predictions and observations |
| Correlation | Measures agreement in temperature variations |

Evaluation should assess performance by depth, region, season, and relevant ocean conditions.

Metric values must be reported only after the corresponding experiments have been completed.

**Validation safeguard:** If GLORYS or another gridded analysis is used to create training targets, it should not be described as independent in-situ ground truth. Argo validation profiles should be kept separate from training and model selection wherever possible.

## 2.7 Overall Workflow

1. Dataset collection
2. Dataset audit and quality checking
3. Preprocessing and harmonization
4. Data integration and split design
5. Satellite embedding
6. Deep learning reconstruction
7. Depth-wise temperature field generation
8. Independent validation and evaluation

### Workflow Diagram

```text
Dataset Collection
        ↓
Audit and Quality Checking
        ↓
Preprocessing and Harmonization
        ↓
Data Integration and Split Design
        ↓
Satellite Embedding
        ↓
Deep Learning Reconstruction
        ↓
Depth-wise Temperature Fields
        ↓
Independent Validation and Evaluation
```

---

# 3. Datasets

The following sections document the datasets identified for OceanEmbed. Exact roles, versions, selected time ranges, downloaded files, and preprocessing decisions must be confirmed against the data actually used in the implementation.

> **Dataset audit note:** The details below are based on the supplied project documentation. Confirm product metadata, temporal coverage, access links, and variables against official provider documentation before treating them as final. Blank fields from the original documentation are marked as pending rather than filled with assumed values.

## 3.1 OISST Dataset

### 3.1.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | NOAA 1/4° Daily Optimum Interpolation Sea Surface Temperature |
| Acronym | OISST |
| Version | Version 2.1 |
| Provider | NOAA / National Centers for Environmental Information (NCEI) |
| DOI | [10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57) |
| Official Source | [NOAA NCEI](https://www.ncei.noaa.gov/) |
| Raw Format | NetCDF / NetCDF-4 |

### 3.1.2 Temporal Coverage

| Field | Details |
|---|---|
| Reported start date | 1981-09-01 |
| End date | Present; verify against the selected product |
| Selected project time range | Pending confirmation |
| Temporal resolution | Daily |
| Sampling cadence | 1 day |

### 3.1.3 Spatial Characteristics

| Field | Details |
|---|---|
| Spatial coverage | Global |
| Grid | 0.25° × 0.25° |
| Project region | 5°N–30°N, 45°E–105°E |
| Vertical coverage | Sea surface |

### 3.1.4 Variables

| Field | Details |
|---|---|
| Variables listed in source documentation | `sst`, `anom`, `err`, `ice` |
| Units listed | SST, anomaly and error: °C; ice: fraction |
| Variables actually used by model | Pending confirmation |

### 3.1.5 Acquisition and Storage

| Field | Status |
|---|---|
| Acquisition source | NOAA / NCEI |
| Acquisition script and command | Pending documentation |
| Acquisition date and time | Pending documentation |
| Retry/failure notes | Pending documentation |
| Estimated raw data size | Pending measurement |
| Number of raw files | Pending measurement |
| Repository/data storage path | Pending documentation |

### 3.1.6 Processing and Quality Control

The supplied documentation describes OISST as a quality-controlled optimum-interpolation product with bias correction.

| Field | Status |
|---|---|
| Processing and standardization steps | Pending implementation details |
| Missing-data behavior | Verify from product metadata and files |
| Quality-control flags and rules applied | Pending documentation |
| Known limitations and biases considered | Pending documentation |
| Final output dimensions and size | Pending measurement |

### 3.1.7 Integration in OceanEmbed

| Field | Details |
|---|---|
| Intended purpose | Sea surface temperature input |
| Model stage | Surface-input / embedding stage |
| Fusion with | SSS, SSH/SLA, currents and winds |
| Spatial/temporal alignment method | Pending implementation details |

### 3.1.8 Selection Decision and Reproducibility

**Reason for selection:** Provides long-term daily SST at the required 0.25° resolution.

Record the following after acquisition and processing:

- Important selection decisions and alternatives considered
- Audit findings during data inspection and collection
- Exact download endpoint or query
- Checksums or file-validation results
- Repository path or data-manifest reference
- Provider documentation URL and audit evidence

## 3.2 GlobCurrent Dataset

### 3.2.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | Global Total (COPERNICUSGLOBCURRENT), Ekman and Geostrophic Currents at Surface and 15 m |
| Acronym | GlobCurrent |
| Provider | Copernicus Marine Service |
| Product ID as recorded | `MULTIOBS GLO PHY MYNRT 015 003` |
| DOI | [10.48670/mds-00327](https://doi.org/10.48670/mds-00327) |
| Raw Format | NetCDF-4 |
| Version/edition | Pending confirmation |

### 3.2.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start date | 1993-01-01 |
| End date | Present; verify against the selected product |
| Selected project time range | Pending confirmation |
| Temporal resolution | Source documentation lists hourly/daily/monthly; verify selected product |
| Spatial coverage | Global |
| Reported grid | 0.25° × 0.25° |
| Reported latitude range | Approximately −89.87° to 89.88° |
| Reported longitude range | Approximately −179.87° to 179.88° |
| Vertical representation | Surface and 15 m |

### 3.2.3 Variables and Units

Variables listed in the source documentation:

- Eastward and northward sea-water velocity
- Ekman and geostrophic current components
- Units: m/s

Exact variable names and variables used by the model remain pending confirmation.

### 3.2.4 Acquisition, Processing and Quality Control

| Field | Status |
|---|---|
| Acquisition source | Copernicus Marine Service |
| Acquisition script/date/time | Pending documentation |
| Retry/failure notes | Pending documentation |
| Number and size of raw files | Pending measurement |
| Processing and standardization | Pending implementation details |
| Missing-data behavior and QC | Pending verification |
| Final output dimensions | Pending measurement |

### 3.2.5 Integration and Audit

**Intended purpose:** Candidate surface ocean-current input.

Fusion with other inputs, alignment method, model stage, selection rationale, alternatives, audit findings, checksums, exact download query, repository path, and provider evidence should be recorded after the selected product and files have been verified.

## 3.3 Argo Dataset

### 3.3.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | Argo Float Data and Metadata |
| Acronym | Argo |
| Provider | International Argo Program / Argo Global Data Assembly Centres (GDACs) |
| Data Access | [Argo Data Access](https://argo.ucsd.edu/data/data-access/) |
| DOI | [10.17882/42182](https://doi.org/10.17882/42182) |
| Raw Format | NetCDF |

### 3.3.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start | Approximately 2000 |
| End date | Present; verify the downloaded archive |
| Selected project time range | Pending confirmation |
| Temporal sampling | Profile/float-cycle dependent; typically about 10 days per float cycle |
| Spatial coverage | Global ocean |
| Spatial representation | Irregular point profiles |
| Vertical coverage | Surface to deep-ocean profiles |

### 3.3.3 Variables and Units

| Variable | Description / Units |
|---|---|
| Pressure | dbar |
| Temperature | °C |
| Salinity | Practical Salinity Scale / PSU as represented by product |
| QC flags | Quality-control flags supplied with profile measurements |
| Adjusted temperature | Temperature adjusted according to Argo processing |
| Adjusted salinity | Salinity adjusted according to Argo processing |

**Variables actually used:** Pending confirmation. Document whether raw or adjusted measurements are used and which QC flags are accepted.

### 3.3.4 Acquisition, Processing and Quality Control

Argo profiles are obtained through Argo GDACs. The source data provide quality-control flags. Irregular spatial and temporal sampling is a key limitation.

| Field | Status |
|---|---|
| Acquisition method/script/date | Pending documentation |
| Files collected and total size | Pending measurement |
| Profile filtering and QC rules | Pending implementation details |
| Missing-data treatment | Pending implementation details |
| Depth interpolation or matching method | Pending implementation details |
| Repository/data manifest path | Pending documentation |

### 3.3.5 Integration in OceanEmbed

**Primary intended purpose:** Independent reference and validation of reconstructed subsurface temperature.

Argo profiles should be matched to predictions by location, time, and depth. The project should document matching tolerances, QC filters, interpolation methods, and exclusion rules.

Validation profiles should be held out from training and model selection when independent evaluation is claimed.

### 3.3.6 Selection Decision and Reproducibility

**Reason for selection:** Provides in-situ subsurface ocean temperature profiles for validation.

Record the selected time range, geographic filtering, exact download method, data version, QC decisions, profile counts, checksums or manifest, and audit evidence.

## 3.4 CCMP Dataset

### 3.4.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | RSS CCMP 6-Hourly 10 Meter Surface Winds Level 4 |
| Acronym | CCMP |
| Version | Version 3.1 |
| Provider | Remote Sensing Systems (RSS) |
| Product ID as recorded | `CCMP WINDS 10M6HR L4 V3.1` |
| DOI | [10.5067/CCMP-6HW10M-L4V31](https://doi.org/10.5067/CCMP-6HW10M-L4V31) |
| Raw Format | NetCDF-4 |

### 3.4.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start date | 1993-01-01 |
| End date | Present; verify against selected product |
| Selected project time range | Pending confirmation |
| Temporal resolution | 6-hourly |
| Sampling cadence | Four observations per day |
| Reported file cadence | One NetCDF-4 file per day |
| Spatial coverage | Near-global |
| Grid | 0.25° × 0.25° |
| Latitude range | 80°S to 80°N |
| Longitude range | −180° to 180° |
| Vertical representation | 10 m surface wind |

### 3.4.3 Variables and Units

| Variable | Description / Units |
|---|---|
| `uwnd` | Wind component, m/s |
| `vwnd` | Wind component, m/s |
| `ws` | Wind speed, m/s |
| `nobs` | Observation-count information |
| `latitude`, `longitude`, `time` | Coordinates and time |

Variables actually used by the model remain pending confirmation.

### 3.4.4 Acquisition, Processing and Quality Control

The source documentation identifies RSS as the acquisition source and notes that observation-count information is provided through `nobs`.

The product's near-global coverage does not extend beyond approximately 80° latitude, which includes the project's stated latitude range.

Acquisition script, dates, file count, data size, missing-data handling, processing steps, output dimensions, and quality-control decisions remain to be documented.

### 3.4.5 Integration and Selection Decision

**Intended purpose:** Surface wind input for the embedding stage, fused with SST, SSS, SSH/SLA, and surface currents.

The project documentation records CCMP as the selected wind dataset instead of ERA5. Record the exact reason for this decision, selected time range, time aggregation method, and validation evidence.

## 3.5 GLORYS Dataset

### 3.5.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | CMEMS Global Ocean Ensemble Physics Reanalysis |
| Acronym | GLORYS |
| Provider | Copernicus Marine Service |
| Product ID as recorded | `GLOBAL MULTIYEAR PHY ENS 001 031` |
| Version/edition | Pending confirmation |
| DOI/citation | Pending confirmation |
| Raw Format | NetCDF-4 |

### 3.5.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start year | 1993 |
| Reported end year | 2024 |
| Selected project time range | Pending confirmation |
| Temporal resolution | Source documentation lists daily/monthly; verify selected product |
| Spatial coverage | Global ocean |
| Reported grid | 0.25° × 0.25° |
| Vertical coverage | 75 vertical levels |

### 3.5.3 Variables and Units

- `thetao`: Potential temperature, °C as recorded in the project notes
- `so`: Salinity, PSU as recorded
- `uo`: Eastward velocity, m/s
- `vo`: Northward velocity, m/s
- Sea surface height and other ocean-physics variables

Variables actually used remain pending confirmation.

### 3.5.4 Acquisition, Processing and Quality Control

| Field | Status |
|---|---|
| Acquisition script and date | Pending documentation |
| Selected product/version | Pending verification |
| Selected time range | Pending confirmation |
| File count and storage size | Pending measurement |
| Preprocessing steps | Pending implementation details |
| Missing-data behavior | Pending verification |
| Quality-control decisions | Pending documentation |
| Final array dimensions | Pending measurement |

### 3.5.5 Integration in OceanEmbed

**Potential purpose:** Ocean-state information and subsurface reference or target fields.

If GLORYS is used as a training target, the model should be described accurately as learning to reconstruct GLORYS-consistent temperature fields unless independent observations establish broader accuracy.

Evaluation against independent Argo profiles is important to avoid treating a reanalysis-derived target as independent truth.

Record the exact variables and depths extracted, interpolation or regridding methods, data-split strategy, and dataset provenance.

## 3.6 OSCAR Dataset

### 3.6.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | Ocean Surface Current Analyses Real-time (OSCAR) Surface Currents — Final 0.25 Degree |
| Acronym | OSCAR |
| Version | Version 2.0 |
| Provider | NASA PO.DAAC / Earth & Space Research (ESR) |
| Product ID as recorded | `OSCAR L4 OC FINAL V2.0` |
| DOI | [10.5067/OSCAR-25F20](https://doi.org/10.5067/OSCAR-25F20) |
| Raw Format | NetCDF-4 |

### 3.6.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start date | 1993-01-01 |
| End date recorded in source notes | 2026-01-17; verify against current product |
| Selected project time range | Pending confirmation |
| Temporal resolution | Daily |
| Spatial coverage | Global |
| Grid | 0.25° × 0.25° |
| Latitude range | −89.75° to 89.75° |
| Longitude range | −180° to 180° |
| Vertical representation | Surface current representative of the upper approximately 30 m mixed layer |

### 3.6.3 Variables and Units

Variables listed in the source documentation:

- `u`, `v`
- `ug`, `vg`

**Units:** m/s.

The exact variables used by the model remain to be confirmed.

### 3.6.4 Acquisition, Processing and Quality Control

The source documentation lists NASA PO.DAAC as the acquisition source. Final, interim, and near-real-time product levels are noted as available; the selected product level must be specified.

The source notes that surface current estimates are derived using physical relationships involving sea surface height, winds, and SST gradients. This should be considered when evaluating overlap with other input datasets.

Acquisition scripts, dates, file counts, storage size, preprocessing, missing-data behavior, quality-control rules, and final dimensions remain to be documented.

### 3.6.5 Integration and Selection Decision

**Intended purpose:** Surface ocean-current input.

The source documentation lists GlobCurrent as an alternative. The project should state which product is actually used, whether OSCAR and GlobCurrent are both retained, and why. Avoid duplicating strongly overlapping inputs without a documented reason.

## 3.7 OISSS Dataset

### 3.7.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | Multi-Mission Optimally Interpolated Sea Surface Salinity Global Dataset V2 |
| Acronym | OISSS |
| Version | Version 2 |
| Provider | NASA / Jet Propulsion Laboratory (JPL) / PO.DAAC |
| Product ID as recorded | `OISSS L4 multimission 7day v2` |
| DOI/citation | Pending confirmation |
| Raw Format | NetCDF-4 |

### 3.7.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start date | 2011-08-24 |
| End date | Present; verify against selected product |
| Selected project time range | Pending confirmation |
| Temporal resolution recorded | 4-day; verify against product metadata and product ID |
| Spatial coverage | Global |
| Grid | 0.25° × 0.25° |
| Vertical coverage | Sea surface |

### 3.7.3 Variables and Units

Variables listed in the source documentation:

- `sss`
- Sea surface salinity empirical uncertainty
- Sea surface salinity formal uncertainty

The source notes list salinity units as \(10^{-3}\). Confirm the official product units and scaling before preprocessing.

Variables actually used by the model remain pending confirmation.

### 3.7.4 Acquisition, Processing and Quality Control

The source documentation lists NASA PO.DAAC / Earthdata as the acquisition source and notes that empirical and formal uncertainty variables are available.

Acquisition script, download date, file count, size, preprocessing, missing-data handling, output dimensions, and quality-control decisions remain to be documented.

### 3.7.5 Integration and Selection Decision

**Intended purpose:** Sea surface salinity input, fused with SST, SSH/SLA, currents, and winds.

The project documentation records Version 2 as the selected product. Record the exact product version, cadence, temporal alignment method, and alternatives considered.

### 3.7.6 Open Questions

- Confirm whether the selected product is four-day or seven-day.
- Confirm the official DOI and landing page.
- Confirm actual variable names, units, scale factors, and uncertainty usage.
- Document the interpolation or temporal alignment method used to create daily inputs.

## 3.8 DUACS Dataset

### 3.8.1 Dataset Identity

| Field | Details |
|---|---|
| Dataset Name | Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed |
| Acronym | DUACS |
| Edition recorded | DT-2024 |
| Provider | Copernicus Marine Service / Mercator Ocean International |
| Product ID | `SEALEVEL_GLO_PHY_L4_MY_008_047` |
| DOI | [10.48670/moi-00148](https://doi.org/10.48670/moi-00148) |
| Official Source | [Copernicus Marine product page](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description) |
| Raw Format | NetCDF-4 |

### 3.8.2 Temporal and Spatial Characteristics

| Field | Details |
|---|---|
| Reported start year | 1993 |
| End date | Present; verify selected product |
| Selected project time range | Pending confirmation |
| Temporal resolution | Daily |
| Spatial coverage | Global |
| Original grid recorded | 0.125° × 0.125° |
| Vertical coverage | Surface variables |

### 3.8.3 Variables and Units

| Variable / Quantity | Units |
|---|---|
| Sea Surface Height | m |
| Sea Level Anomaly (SLA) | m |
| Absolute Dynamic Topography (ADT) | m |
| Geostrophic eastward velocity | m/s |
| Geostrophic northward velocity | m/s |

Exact variable names and selected fields remain pending confirmation.

### 3.8.4 Acquisition, Processing and Quality Control

The source documentation identifies Copernicus Marine Service as the acquisition source.

The original grid is recorded as 0.125°, while the project target grid is 0.25°. The method used to harmonize these grids must be recorded.

Acquisition scripts, date/time, file counts, data size, processing steps, missing-data behavior, quality-control rules, output dimensions, and checksums remain to be documented.

### 3.8.5 Integration and Selection Decision

**Intended purpose:** Surface sea-level and current-related input.

The project should document which variables are used, whether geostrophic velocities overlap with other current datasets, how regridding is performed, and how data are aligned with daily model inputs.

---

# 4. Validation and Reproducibility

Use this checklist to complete the documentation with evidence from the actual implementation. Do not mark an item complete until the corresponding code, data, or experiment has been verified.

## 4.1 Dataset Acquisition

- [ ] Record official product IDs, versions, DOI/landing pages, and license or access terms.
- [ ] Record the exact selected time range for each dataset.
- [ ] Record the acquisition date and download method/script.
- [ ] Record the number of files, storage size, and data manifest.
- [ ] Preserve relevant provider metadata and citation requirements.

## 4.2 Preprocessing and Integration

- [ ] Document coordinate conventions and region cropping.
- [ ] Document regridding from source grids to 0.25° × 0.25°.
- [ ] Document daily alignment for six-hourly, four-day/seven-day, monthly, and irregular profile data.
- [ ] Document units, scale factors, variable names, and missing-value handling.
- [ ] Document quality-control rules and excluded observations.
- [ ] Document normalization statistics and ensure they are fitted on training data only.
- [ ] Document treatment of missing inputs and masked land/ocean cells.
- [ ] Record final input and target array shapes.

## 4.3 Model Development

- [ ] Document the embedding architecture and how embeddings are generated.
- [ ] Document the reconstruction model, input channels, target depths, and output dimensions.
- [ ] Record training configuration, random seeds, software versions, and hardware.
- [ ] Record training, validation, and test split strategy.
- [ ] Prevent spatial and temporal leakage between data splits.
- [ ] Save model checkpoints and configuration files where appropriate.

## 4.4 Validation and Reporting

- [ ] Define the independent validation protocol using Argo profiles.
- [ ] Specify QC flags, location/time matching tolerances, and depth interpolation.
- [ ] Report RMSE, MAE, bias, and correlation with units and sample counts.
- [ ] Report metrics by depth and relevant geographic or temporal group.
- [ ] Compare against simple baselines where feasible.
- [ ] Distinguish validation against independent observations from agreement with a reanalysis product.
- [ ] Include plots and tables generated from actual evaluation results.
- [ ] Do not invent metric values or claim experiments that have not been completed.

## 4.5 Reproducibility and Audit Trail

- [ ] Record repository paths for scripts, configurations, manifests, and model outputs.
- [ ] Record file checksums or other integrity checks where feasible.
- [ ] Document unresolved issues and known limitations.
- [ ] Keep credentials, access tokens, and private data out of the repository.
- [ ] Add citations and acknowledgements required by dataset providers.

---

# 5. Limitations and Important Considerations

## 5.1 Indirect Reconstruction

Surface satellite observations do not uniquely determine the full subsurface temperature profile. The learned relationships may be uncertain, particularly at greater depths or in regions with limited observations.

## 5.2 Dataset Coverage and Compatibility

The selected products have different time ranges, grids, temporal cadences, and quality-control conventions. Only the overlapping and appropriate data should be integrated, and every transformation should be documented.

## 5.3 Reanalysis Versus Independent Observations

If GLORYS is used to supply training targets, predictions may learn the characteristics and biases of that reanalysis. Independent Argo observations are therefore important for evaluating how well reconstructed temperatures agree with in-situ measurements.

## 5.4 Missing Data and Uncertainty

Cloud-related gaps, satellite coverage limitations, irregular Argo sampling, and product-specific uncertainties can affect model performance. Missing-data handling and uncertainty assessment should be reported transparently.

## 5.5 Current Dataset Status

A dataset listed in this report is not automatically evidence that it has been downloaded, integrated, or used for training. Likewise, proposed model components and validation procedures should not be presented as completed results until supported by implementation evidence.

Fields marked **Pending confirmation**, **Pending documentation**, or **Pending measurement** should be completed using verified metadata and results from the project repository.

---

## Project Credits

**TARAL — Thermal Analysis and Reconstruction using AI for Ocean Layers**  
**OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

**Team:** B2S — Bharat Through Space  
**Event:** Smart India Hackathon 2026  
**Problem Statement ID:** 26066  
**Category:** Software  
**Theme:** Disaster Management

