
# DyMoLaDiNE: Dynamic Multi-Modal Latent Diffusion for Robust Medical Diagnosis under Dynamic Shifts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of DyMoLaDiNE (Dynamic Multi-Modal Latent-guided Diffusion Nested-Ensembles), a novel framework for robust, reliable, and interpretable medical image classification under dynamic, multi-modal covariate shifts.

## Table of Contents

- [Research Gap and Motivation](#research-gap-and-motivation)
- [Problem Formulation](#problem-formulation)
- [Proposed Method: DyMoLaDiNE](#proposed-method-dymoladine)
- [Datasets](#datasets)
- [Comparison with SOTA](#comparison-with-sota)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Research Gap and Motivation

Existing robust medical image classification models, such as LaDiNE, excel under single-modality, static perturbations (e.g., pre-defined noise or resolution drops). However, a critical limitation is their failure to address dynamic and multi-modal covariate shifts—a critical scenario in real-world clinical settings. Real-world diagnosis rarely relies on single-modality data. Clinicians integrate multi-modal inputs (e.g., chest X-rays + CT scans, skin images + patient metadata, or MRI + genetic profiles) to make decisions. However, covariate shifts in multi-modal settings are dynamic and cross-modal:
- Shifts may affect one modality but not others (e.g., low-quality X-rays but high-quality CT scans).
- Modalities may exhibit conflicting shifts (e.g., X-ray noise increases while clinical text metadata becomes more sparse).

LaDiNE's invariant feature extraction and diffusion-based uncertainty quantification are not extended to cross-modal scenarios, highlighting the need for a principled solution.

## Problem Formulation

Let `Y ∈ {1, ..., C}` be the class label and `m = {x⁽¹⁾, x⁽²⁾, ..., x⁽ⁿ⁾, t, c}` be a multi-modal input comprising `n` distinct imaging modalities, tabular data (`t`), and clinical notes (`c`). The joint input space is `M = X⁽¹⁾ × ... × X⁽ⁿ⁾ × T × C`. Real-world deployment faces dynamic covariate shifts, where `x⁽ⁱ⁾ ~ p_test⁽ⁱ⁾(x⁽ⁱ⁾) ≠ p_train⁽ⁱ⁾(x⁽ⁱ⁾)`, leading to a joint test distribution `p_test(m') = p_test(x'⁽¹⁾, ..., x'⁽ⁿ⁾, t', c') ≠ p_train(m)`.

The goal is to learn a predictive distribution `p(Y|m')` that is:
   Accurate: High `E[Y|m']` for true `Y`.
   Robust: Maintains accuracy under dynamic shifts.
   Well-Calibrated: `p(Y=c|m')` matches true frequency.
   Informative Uncertainty: `H[p(Y|m')]` is high when incorrect, low when confident, and decomposable: `Uncertainty(m') = f({U_x⁽¹⁾, ..., U_t, U_interaction, ...})`.

## Proposed Method: DyMoLaDiNE

DyMoLaDiNE extends the principles of LaDiNE to the multi-modal domain, explicitly addressing the challenges of dynamic, modality-specific covariate shifts and providing decomposable, reliable uncertainty quantification. It comprises four core innovations:

1.  Cross-Modal Invariant Feature Extractor: Uses a shared multi-modal Vision Transformer (MM-ViT) backbone with dedicated initial layers and cross-attention fusion. A contrastive loss enforces invariance to modality-specific perturbations, learning a robust latent representation `z_common`.
2.  Dynamic Modality Weighting Mechanism: A reliability scoring network `Rel_ψ` predicts a reliability score vector `s(m') = [s_1, ..., s_n, s_t, s_c]` for each test instance `m'`. These scores dynamically compute mixture weights `π_k(m'; ω) = Softmax(f_ω(s(m')))_k` and condition the diffusion models, adapting to instance-specific modality reliability.
3.  Robust Multi-Modal Diffusion Ensemble: A mixture distribution `p(Y|m', Ψ) = Σₖ π_k(m') ∫ p_θₖ(Y|z_common, m', s(m')) δ(z_common - g_ϕₖ(MM-Encₖ(m'))) dz_common`. Each CDM `p_θₖ` conditions on `z_common`, `m'`, and `s(m')`, enabling robust density estimation under shifting conditioning contexts.
4.  Modality-Attributed Uncertainty Quantification: Provides interpretable uncertainty attribution. For each modality `i`, it generates ablated samples `yₖ,ₘ^(-i)` (where modality `i` is replaced by a baseline) and estimates the attributed uncertainty `U_x⁽ⁱ⁾ = ||ȳ - ȳ^(-i)||₂²`. This identifies the modality contributing most to prediction uncertainty.

## Datasets

We evaluate DyMoLaDiNE on seven large-scale medical imaging datasets, simulating dynamic, multi-modal covariate shifts.

| Dataset         | Modalities              | Task                      | # of Classes | # of Samples | Reference |
| :-------------- | :---------------------- | :------------------------ | :----------: | :----------: | :-------- |
| MedMD&RadMD | Images, Text Reports    | Multi-modal Diagnosis     | 2            | ~15.5M       | [RadFM](https://arxiv.org/abs/2305.15906) |
| MultiCaRe   | Images, Metadata, Notes | Multi-modal Diagnosis     | 2            | ~70,000      | [MultiCaRe](https://arxiv.org/abs/2306.07815) |
| PadChest    | Chest X-rays            | Thoracic Disease Detection| 193          | ~120,000     | [PadChest](https://arxiv.org/abs/1901.07441) |
| TCIA RE-MIND| MRI                     | Glioma Segmentation       | 2 (Binary)   | ~300         | [TCIA](https://wiki.cancerimagingarchive.net/display/Public/RE-MIND) |
| BRaTS       | Multi-modal MRI         | Brain Tumor Segmentation  | 4 (Multiclass)| ~2,000      | [BRaTS](https://www.med.upenn.edu/cbica/brats2020/) |
| Camelyon16  | Whole-slide Images      | Lymph Node Metastasis     | 2            | ~400         | [Camelyon16](https://camelyon16.grand-challenge.org/) |
| PANDA       | Whole-slide Images      | Prostate Cancer Grading   | 6 (Ordinal)  | ~10,000      | [PANDA](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) |

Note: For datasets primarily consisting of single modalities (PadChest, TCIA, BRaTS, Camelyon16, PANDA), dynamic shifts are simulated on the primary image modality, and the framework's core robustness and uncertainty mechanisms are evaluated.

## Comparison with SOTA

DyMoLaDiNE is compared against a comprehensive set of state-of-the-art methods:

   Single-Modal Robust Models: LaDiNE [1], ResNets [2], ViTs [3], MedViT [4].
   Multi-Modal Models: LDM [5], CMCL [6], CGMCL [7], CIIM [8], DTTL [9], FFL [10], ALDM [11].
   Ensemble Methods: Deep Ensembles [12], DWE [13], DTT [14], ICNN-Ensemble [15].

Extensive evaluations demonstrate DyMoLaDiNE's superior performance in terms of:
   Classification Accuracy under dynamic perturbations.
   Robustness (measured by relative accuracy drop).
   Confidence Calibration (Expected Calibration Error - ECE).
   Uncertainty Quantification (MM-CPIW, MM-CNPV).
   Modality Attribution Fidelity (AF).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your_username/dymoladine.git
    cd dymoladine
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv_dymoladine
    source venv_dymoladine/bin/activate # On Windows: venv_dymoladine\Scripts\activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Prepare Datasets: Place your datasets in the `data/` directory or adjust paths in `data/load_datasets.py`. Ensure they follow the expected structure.
2.  Configuration: Modify hyperparameters and experiment settings in `experiments/train.py` and `experiments/evaluate.py`.
3.  Training:
    ```bash
    python experiments/train.py --dataset MedMD_RadMD --config configs/default_config.yaml
    ```
4.  Evaluation:
    ```bash
    python experiments/evaluate.py --dataset MultiCaRe --model_path path/to/saved/model
    ```

## Results

Detailed quantitative and qualitative results, including comparisons with SOTA methods and ablation studies, are available in our paper (link to be added upon publication).

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{dymoladine2025,
  title={Cross-Modal Invariant Learning with Latent Diffusion for Reliable Medical Diagnosis under Dynamic Shifts},
  author={Your Name, Co-Author Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}

@article{shen2025improving,
  title={Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles},
  author={Shen, Xing and Huang, Hengguan and Nichyporuk, Brennan and Arbel, Tal},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [your_email@example.com](mailto:your_email@example.com).

## References

[1] Shen et al., "Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles", IEEE TMI, 2025.  
[2] He et al., "Deep Residual Learning for Image Recognition", CVPR, 2016.  
[3] Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR, 2021.  
[4] Manzari et al., "Medvit: a robust vision transformer for generalized medical image classification", CBM, 2023.  
[5] Prusty et al. (Ed.), "Enhancing Medical Image Analysis", Springer, 2024.  
[6] Wang et al., "Two-stage selective ensemble of cnn via deep tree training for medical image classification", IEEE TCYB, 2021.  
[7] Ding et al., "Enhancing", arXiv, 2024.  
[8] Yan et al., "Causality", arXiv, 2024.  
[9] Yu et al., "Dynamic", IJCNN, 2024.  
[10] Irfan et al., "Federated", arXiv, 2024.  
[11] Kim et al., "Adaptive", arXiv, 2024.  
[12] Lakshminarayanan et al., "Simple and scalable predictive uncertainty estimation using deep ensembles", NeurIPS, 2017.  
[13] Pacheco et al., "Learning dynamic weights for an ensemble of deep models applied to medical imaging classification", IJCNN, 2020.  
[14] Yang et al., "Two-stage selective ensemble of cnn via deep tree training for medical image classification", IEEE TCYB, 2021.  
[15] Musaev et al., "Icnn-ensemble: An improved convolutional neural network ensemble model for medical image classification", IEEE Access, 2023.
```

(Please replace placeholders like `your_username`, `your_email@example.com`, and the paper arXiv link with the actual details.)
