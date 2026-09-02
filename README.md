# TRUE-Net-Coastal-Inversion
Probabilistic Satellite Retrievals of Coastal Chlorophyll via Prior-Informed Deep Learning and Uncertainty Partitioning
# TRUE-Net: Trust-Region Uncertainty Ensemble Network

This repository contains the core Python source code for the Trust-Region Uncertainty Ensemble Network (TRUE-Net), an architecture designed for probabilistic satellite retrievals of coastal Chlorophyll-$\alpha$. This codebase accompanies the manuscript submitted to *Remote Sensing of Environment* (RSE).

TRUE-Net utilizes a deterministic convolutional core to generate continuous Lognormal Probability Density Functions (LN PDFs). It dynamically partitions total predictive variance into its aleatory (environmental volatility) and epistemic (model/sensor ignorance) components using a continuous temporal routing parameter ($t_{mod}$) and a Spectral Trust Index ($A_{spec}$).

## Data Availability & Download Instructions

Due to GitHub's storage constraints, the large spatial tensors and multi-year data arrays required to execute this code are hosted separately on Zenodo. 

**Zenodo Archive:** [doi.org/10.5281/zenodo.22249365]

Before running the scripts, you must download the necessary datasets from the Zenodo archive. The archive includes:
*   Multi-model stochastic initializations (`Multi_Realization_20m`)
*   Training tensors (`prepared_dat_58x20`)
*   Full basin validation tensors (`prepared_xlarge`)
*   Spatial matrices for the 22 sequential Sentinel-2 overpasses (`Sentinel-2_22overpasses`)

## Crucial: Local Directory Setup

**For the proper execution of all scripts, you must configure your local directory paths.** 

Depending on where you extract the downloaded Zenodo files on your machine, you must update the input/output directory paths inside the scripts (or within the `TRUE_config.py` file, if utilized). 

1. Create dedicated folders on your local machine for the input tensors, the model realizations, and the output prediction matrices.
2. Extract the Zenodo downloads into their respective input folders.
3. Open each Python script and verify that the defined `data_path`, `model_save_path`, and `output_path` variables correctly point to your newly created local directories. 

## Execution Sequence

The TRUE-Net pipeline is divided into three distinct stages. The scripts must be executed in the exact sequence below.

### Step 1: Training the Ensemble
**Script:** `Train_Seasonal_TRUE_Net_v1_0.py`
This script trains the $M=5$ independent deep learning ensemble members over the localized estuarine test polygon. It utilizes the prior-informed bio-optical regularization ($\lambda=0.1$) and the continuous harmonic temporal embedding to optimize the network across the annual cycle. The converged model weights will be saved to your designated realizations directory.

### Step 2: Full-Basin Inference and Uncertainty Partitioning
**Script:** `Infer_MultiRegime_TRUE_Net_v1_0.py`
This script loads the trained ensemble weights and applies them to the full macro-basin spatial tensors across the 22 Sentinel-2 overpasses. During inference, it evaluates the Spectral Trust Index ($A_{spec}$) pixel-by-pixel, triggers the Truncated Gaussian Umbrella to estimate aleatory uncertainty ($\sigma_{spec}$), and calculates epistemic variance from the ensemble divergence. 

### Step 3: Visualization and Probabilistic Distribution
**Script:** `Plot_TRUE_Net_Publication_Figures_v1_0.py`
This script ingests the output prediction matrices generated in Step 2 to recreate the figures presented in the manuscript. It synthesizes the spatial and partitioned uncertainties to generate the final Lognormal Probability Density Functions (LN PDFs), the regional chronological time series, and the 2x2 spatial uncertainty mapping panels.

## Dependencies
* Python 3.8+
* PyTorch (or TensorFlow, depending on your backend)
* NumPy, Pandas
* Matplotlib, Seaborn
* SciPy
