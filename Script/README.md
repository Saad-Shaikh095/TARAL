# Script Directory

This directory contains all Python scripts and organized code for the TARAL OceanEmbed project, including data acquisition, model training, auditing, and software package development.

## Directory Structure

### 📁 All_py_files/
Collection of standalone Python scripts for data fetching and auditing:

- **fetch_CMEMS_DUACS.py** - Fetch CMEMS DUACS (Data Unification and Altimeter Combination System) data
- **fetch_CMEMS_GlobCurrent.py** - Fetch CMEMS Global Ocean Current data
- **fetch_GLORYS.py** - Fetch GLORYS (Global Ocean Reanalysis and Simulations) data
- **fetch_OISSS.py** - Fetch OISSS (Optimum Interpolation Sea Surface Salinity) data
- **fetch_OISST.py** - Fetch OISST (Optimum Interpolation Sea Surface Temperature) data
- **glorys_scientific_audit.py** - Scientific audit script for GLORYS data
- **glorys_scientific_audit_1.py** - Alternative version of GLORYS audit script

### 📁 All_OceanEmbed_Scripts/
Main organized repository of OceanEmbed scripts, structured into functional categories:

#### 🔬 audits/
Comprehensive collection of data validation and auditing scripts (35+ files):

**Argo Profile Audits:**
- `argo_audit.py`, `argo_scientific_audit.py` - Main Argo data audits
- `audit_argo.py`, `audit_argo2.py` - Additional Argo validation scripts
- `audit_argo_index.py`, `audit_argo_profiles.py` - Index and profile-specific audits
- `check_argo_manifest.py`, `scan_argo.py` - Manifest checking and scanning utilities

**Oceanographic Data Audits:**
- `audit_glorys_spatial.py` - GLORYS spatial data validation
- `audit_oscar_native.py`, `oscar_audit.py`, `oscar_audit_2.py` - OSCAR data audits
- `oisss_audit.py`, `oisss_special_audit.py`, `oisss_special_audit_v2.py` - Sea surface salinity audits
- `test_oisst.py` - Sea surface temperature testing

**Current and Wind Data Audits:**
- `globecurrent_audit.py`, `globecurrent_analysis_2.py` - Global ocean current analysis
- `ccmp_acquisition_test.py` - CCMP wind data testing
- `test_duacs.py` - DUACS altimetry data testing

**Comprehensive System Audits:**
- `oceanembed_forensic_audit_final.py` - Final forensic audit
- `oceanembed_forensic_audit_final_patched.py` - Patched version
- `oceanembed_forensic_audit_staged.py` - Staged audit version
- `oceanembed_raw_source_audit.py` - Raw data source validation

**Test and Development:**
- `audit_quick.py` - Quick audit utility
- `test_manifest.py` - Manifest validation
- `test_amit.py`, `test_amit_oisst2011.py` - Custom test scripts
- `test_audit_era5.py` - ERA5 reanalysis data testing
- `test_oscar_opendap.py` - OSCAR OpenDAP testing

#### 🌊 data_acquisition/
Scripts for downloading and processing oceanographic data (19 files):

**Multi-Source Fetchers:**
- `fetch_ARGO.py` - Argo float profile data
- `fetch_CCMP.py` - Cross-Calibrated Multi-Platform wind data
- `fetch_CMEMS_DUACS.py` - CMEMS altimetry data
- `fetch_CMEMS_GlobCurrent.py` - CMEMS ocean current data
- `fetch_GLORYS.py` - GLORYS reanalysis data
- `fetch_OISSS.py` - Sea surface salinity data
- `fetch_OISST.py` - Sea surface temperature data
- `fetch_OSCAR.py` - OSCAR current data

**Specialized Acquisition Scripts:**
- `acquire_argo_2012_2024_v6.py` - Argo data for 2012-2024 period
- `download_argo_index.py` - Argo index download utility
- `ccmp_full_acquisition.py`, `ccmp_full_acquisition_v3_1.py`, `ccmp_full_acquisition_v5_opendap.py` - CCMP full dataset acquisition versions
- `download_oscar_full.py`, `download_oscar_v2_opendap.py` - OSCAR full dataset downloads
- `glorys_aquisition.py` - GLORYS acquisition utility

**Advanced GLORYS Processing:**
- `OceanEmbed_GLORYS_ACQUISITION_FULL_VERTICAL_V4_2_5_100x240_AREA_WEIGHTED_WORKING.py` - Full vertical profile acquisition with area weighting
- `OceanEmbed_GLORYS_ACQUISITION_FULL_VERTICAL_V4_2_6_MULTIYEAR_CONFIGURABLE.py` - Multi-year configurable acquisition

#### 🤖 models/

##### architectures/
Neural network model architectures (6 files):
- `OceanEmbed_B8_MSAA_ConvLSTM_FINAL.py` - Multi-Scale Attention Augmented ConvLSTM (B8)
- `OceanEmbed_H1_B7_PAM_ConvLSTM.py` - Position Attention Module ConvLSTM (H1_B7)
- `OceanEmbed_H1_B7_PAM_ConvLSTM_FINAL.py` - Final version
- `OceanEmbed_H1_B7_PAM_ConvLSTM_FINAL_before_warmstart.py` - Pre-warmstart version
- `OceanEmbed_H1_B7_PAM_ConvLSTM_FINAL_claude.py` - Claude-assisted version
- `OceanEmbed_H1_B7_PAM_ConvLSTM_fixed.py` - Fixed version

##### training/
Model training scripts (11 files):
- `OceanEmbed_B6_Native36_REAL_TRAIN_EXACT_SIH.py` - Native 36-level training with exact SIH
- `OceanEmbed_B6_Native36_REAL_TRAIN_EXACT_SIH_FIXED.py` - Fixed version
- `OceanEmbed_B6_Native36_RESUME_EPOCH2.py`, `OceanEmbed_B6_Native36_RESUME_EPOCH2_FIXED.py` - Training resumption scripts
- `OceanEmbed_B9_RESUME_LATEST_MODEL_SAFE.py` - Safe model resumption
- `OceanEmbed_KaggleTrainer_B6_ConvLSTM_15Epoch_LatestOnlyResume_BestRestore.py` - Kaggle trainer with best model restoration
- `OceanEmbed_KaggleTrainer_V1_PointwiseLinear_B1.py` - Pointwise linear baseline (B1)
- `OceanEmbed_KaggleTrainer_V1_PointwiseMLP_B2.py` - Pointwise MLP baseline (B2)
- `OceanEmbed_KaggleTrainer_V1_SpatialCNN_B3_V3_Trainer_1.0.2.py` - Spatial CNN trainer
- `OceanEmbed_Model_PointwiseMLP_V1_B2.py` - Pointwise MLP model
- `OceanEmbed_Native36_Simple_Baseline_Kaggle.py` - Simple baseline for Kaggle

##### evaluation/
Model evaluation scripts (3 files):
- `OceanEmbed_A2_4_FINAL_FAST_GPU.py` - Fast GPU evaluation
- `OceanEmbed_A2_4_FINAL_FAST_GPU_FIXED.py` - Fixed GPU evaluation
- `OceanEmbed_A2_4_Fixed_GPU.py` - Fixed GPU version

#### 🛠️ tools/

##### maintenance/
- `organize_python_scripts.py` - Script organization utility
- `migration_manifest.json` - Migration tracking manifest
- `logs/` - Maintenance logs directory

#### 📦 software_package/
Structured software package development:

##### dev_scripts/
Development and verification scripts:
- `argo_smoke_test.py` - Argo data smoke tests
- `check_backend.py` - Backend verification
- `check_frontend.py` - Frontend verification
- `glorys_live_verification.py` - Live GLORYS data verification
- `launch_oceanembed.bat` - Windows batch launcher
- `test_oisst_acquisition.py` - OISST acquisition testing

##### tests/
Comprehensive test suite:
- `fixtures/` - Test fixtures
- `integration/` - Integration tests
- `smoke/` - Smoke tests
- `unit/` - Unit tests
- `test_package.py` - Package-level tests

##### src/
Source code:
- `oceanembed/` - Main OceanEmbed package source code

##### legacy/
- `acquisition/` - Legacy acquisition scripts

### 📄 Script.txt
Text file (currently empty) for additional script notes or references.

## Usage Notes

### Data Acquisition
Scripts in `data_acquisition/` fetch oceanographic data from various sources including:
- **CMEMS** (Copernicus Marine Environment Monitoring Service)
- **Argo** float profiles
- **GLORYS** reanalysis
- **OISST/OISSS** optimum interpolation products
- **CCMP** wind data
- **OSCAR** ocean currents

### Model Development
The `models/` directory contains:
- Multiple baseline architectures (B1-B9)
- Advanced ConvLSTM with attention mechanisms
- Training scripts for various configurations
- GPU-optimized evaluation tools

### Quality Assurance
The extensive `audits/` directory ensures data quality through:
- Format validation
- Spatial/temporal completeness checks
- Scientific validity audits
- Cross-source consistency verification

## Project Context

This script collection is part of the TARAL OceanEmbed project, which appears to be focused on:
- Oceanographic data integration and processing
- Deep learning models for ocean state prediction
- Multi-source data fusion from satellite and in-situ observations
- Spatial-temporal modeling using ConvLSTM architectures

## File Organization

Scripts are organized by function:
1. **Data Pipeline**: Acquisition → Audit → Processing
2. **Model Pipeline**: Architecture → Training → Evaluation
3. **Software Development**: Source → Tests → DevOps

For detailed documentation about the project, see the `Documentation/` directory in the project root.
