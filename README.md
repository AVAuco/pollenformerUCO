<div align="center">

# 🌾 PollenFormerUCO

### Vision Transformers for Pollen Grain Classification
#### A systematic backbone benchmark on **UCOPollen**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://richardesp-pollenformer-uco-demo.hf.space/)
[![Paper](https://img.shields.io/badge/Paper-Under_Review-9cf?style=for-the-badge)](#-citation)
[![License](https://img.shields.io/badge/License-MIT-3da639?style=for-the-badge)](#-license)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Backbone](https://img.shields.io/badge/Backbone-DINOv2--ViT--B-264796)
![Inference](https://img.shields.io/badge/Inference-Softmax_%7C_k--NN-264796)
![Test Acc](https://img.shields.io/badge/Test_Accuracy-98.1%25-2ea44f)
![Macro F1](https://img.shields.io/badge/Macro--F1-97.5%25-2ea44f)

<br/>

<a href="https://richardesp-pollenformer-uco-demo.hf.space/">
  <img src="https://img.shields.io/badge/▶%20%20Try%20the%20Live%20Demo-Open%20in%20Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Try the live demo" height="42">
</a>

<br/><br/>

<img src="static/pollen_attention_head7.gif" alt="PollenFormerUCO attention rollout over a pollen grain" width="440"/>

<sub><b>PollenFormerUCO</b> last-layer self-attention, head&nbsp;7 (qualitative visualisation).</sub>

</div>

---

## 🔬 Reviewers — start here

This repository accompanies the manuscript *“Vision Transformers for Pollen Grains Classification:
A Systematic Backbone Benchmark on UCOPollen”* (under review). To make the work easy to assess, a
**fully interactive live demo of the trained model is already online** — no installation, no setup:

<div align="center">

### 👉 **[richardesp-pollenformer-uco-demo.hf.space](https://richardesp-pollenformer-uco-demo.hf.space/)**

</div>

In the demo you can:

- 🖼️ **Upload a microscopy image** (JPG / PNG / WEBP / BMP) or pick a bundled sample.
- 🏷️ Get the **predicted pollen taxon** with a confidence indicator.
- 🤝 Inspect the **top-k nearest neighbours** (with cosine similarity) used for k-NN inference.
- 🔥 Overlay the model's **attention map** as a qualitative visualisation.
- 🗺️ Explore the learned **embedding space** via a 2-D projection of the gallery.

> 📦 **The source code, trained model weights, and the dataset will be made publicly available after
> acceptance of the paper.** This page and the live demo are available now so reviewers can evaluate
> the model's behaviour end-to-end.

---

## 🧩 TL;DR

- **Problem.** Automatic pollen identification underpins aerobiology, environmental monitoring, and
  allergy forecasting — but manual microscopy is slow and error-prone.
- **Method.** We benchmark **eleven backbones** spanning CNN, hybrid-attention and self-supervised
  Vision Transformer families: frozen linear probing first, then full fine-tuning of the strongest
  model of each family under an **identical budget**. **PollenFormerUCO** (DINOv2-ViT-B) is
  fine-tuned with a **hybrid Cross-Entropy + Triplet** objective that projects the CLS token into a
  compact **128-D embedding**, read by a **softmax head** or by **cosine nearest-neighbour** search.
- **Results.** **98.1% test accuracy / 97.5% macro-F1** (cosine 1-NN on 128-D), **97.7% / 97.1%**
  (softmax) across **17 pollen taxa + a NoPollen class**.
- **Why it matters.** Under equal-budget fine-tuning the **training regime, not the architecture**,
  governs closed-set performance, and the compact embedding lets **unseen taxa be added to a
  reference gallery without retraining**.

<details>
<summary><b>📄 Read the full abstract</b></summary>

> Automatic pollen identification is critical for aerobiology, environmental monitoring and allergy
> forecasting. We introduce UCOPollen, a 48,950-image dataset covering 17 pollen taxa and one negative
> class, and benchmark eleven backbones spanning convolutional (CNN), hybrid-attention and
> self-supervised Vision Transformer (ViT) families. Phase I compares frozen representations by linear
> probing. Phase II fully fine-tunes the strongest model from each family under an equal budget to test
> whether that ranking persists when optimisation is controlled. Phase II-B adds a hybrid
> Cross-Entropy–Triplet objective and compact projection to select a model for closed-set
> classification and gallery-based recognition of unseen taxa without retraining. Phase III, a post hoc
> analysis on a date-disjoint partition, tests robustness to held-out acquisition dates. Frozen probing
> yields a macro-F1 range (0.668–0.832), led by DINOv2-ViT Base; after equal-budget fine-tuning, the
> spread collapses to 0.0043 and no pairwise difference is detectable (corrected p ≥ 0.247). Thus,
> training regime rather than architecture governs closed-set performance. We introduce
> PollenFormerUCO, the DINOv2-ViT Base hybrid configuration (d = 128, λ = 0.5), with backbone
> parameters trainable under differential learning rates. It reaches 98.1% accuracy and 97.5% macro-F1
> using cosine 1-nearest-neighbour (1-NN) inference on a 128-dimensional embedding. Gallery extension
> lets seven unseen taxa be added and queried without retraining; we evaluate out-of-distribution
> material. Phase III (96.1% accuracy and 88.6% macro-F1 over 17 classes), repeated runs and a
> cross-entropy control delimit these claims and indicate that retrieval capability derives from the
> projection head, not uniquely from the triplet term. The dataset, trained weights, code and online
> demo will be available through the project repository:
> [github.com/AVAuco/pollenformerUCO](https://github.com/AVAuco/pollenformerUCO).

</details>

---

## ✨ Results at a glance

**PollenFormerUCO** — test-set generalization (UCOPollen v1.0):

| Inference method | Test Accuracy | Macro-F1 |
|---|:---:|:---:|
| Softmax head | 97.7% | 97.1% |
| **Cosine 1-NN (128-D embedding)** | **98.1%** | **97.5%** |

- ⚖️ **Training regime, not architecture.** Under frozen probing the eleven backbones span 0.1641
  macro-F1; after equal-budget full fine-tuning the spread collapses to 0.0043 and no pairwise
  difference is detectable.
- 🏆 **DINOv2 under the hybrid objective.** Its softmax head is significantly more accurate than
  ConvNeXt-Base and DaViT-Small trained under the same objective (McNemar p = 1.0×10⁻³ and
  4.2×10⁻⁴), and its 1-NN readout gives the highest macro-F1 of the study.
- 🧲 **Queryable embeddings.** The learned 128-D projection is markedly better separated than the
  768-D CLS token (test silhouette 0.617 → 0.809), so a parameter-free nearest-neighbour rule matches
  the trained head and **new taxa can be added to the gallery without retraining**.

<div align="center">
<img src="static/cm_ce_vs_hybrid.png" alt="Confusion matrices: Cross-Entropy vs. hybrid CE+Triplet" width="92%"/><br/>
<sub>Confusion matrices on the held-out set — Cross-Entropy (left) vs. hybrid CE + Triplet (right).</sub>
</div>

---

## 🧠 Method & architecture

PollenFormerUCO pairs a self-supervised **DINOv2-ViT Base** backbone with a lightweight linear
projection (768 → **128-D**) trained under a **hybrid Cross-Entropy + Triplet** objective. At
inference, a query embedding is classified either by the **softmax head** or by **cosine k-NN** over
a gallery of training embeddings — the latter requires no extra training and exposes interpretable
nearest neighbours.

<div align="center">
<img src="static/architecture_diagram_v2.png" alt="PollenFormerUCO architecture: DINOv2-ViT-B backbone, 128-D projection, metric space, k-NN inference" width="92%"/>
</div>

---

## 📊 Dataset — UCOPollen v1.0

A large, curated collection of **48,950** microscopy images spanning **17 pollen taxa** plus a
**NoPollen** negative class, split in a stratified **70 / 15 / 15** train / val / test partition.

<div align="center">
<img src="static/dataset_distribution.png" alt="UCOPollen class distribution" width="80%"/>
</div>

<div align="center">

| | | | | | |
|---|---|---|---|---|---|
| Brassicaceae | Castanea | Casuarina | Chenopodium | Cupressaceae | Fraxinus |
| Morus | Olea | Pinus | Pistacia | Plantago | Platanus |
| Poaceae | Populus | Quercus | Rumex | Urticaceae | *NoPollen* |

</div>

> 🔒 The UCOPollen dataset will be made **publicly available after acceptance of the paper**.

---

## 🔍 Explainability

The hybrid objective reshapes the feature space into compact, well-separated clusters. The UMAP
projections below contrast the embeddings learned with plain Cross-Entropy versus the hybrid
CE + Triplet objective; the class-centroid view summarizes inter-class separation.

<div align="center">
<img src="static/umap_ce_vs_hybrid.png" alt="UMAP of embeddings: Cross-Entropy vs. hybrid CE+Triplet" width="92%"/><br/>
<sub>UMAP of validation embeddings — Cross-Entropy (left) vs. hybrid CE + Triplet (right).</sub>
<br/><br/>
<img src="static/class_centroids.png" alt="Per-class centroid relationships in the learned embedding space" width="92%"/><br/>
<sub>Class-centroid structure in the learned 128-D embedding space.</sub>
</div>

The UMAP projections and the attention animation at the top are qualitative visualisations only;
every quantitative claim about the embedding rests on separability indices and nearest-neighbour
accuracy computed in the full 128-D space.

---

## 📦 Code, weights & dataset

The **source code, trained model weights, and the UCOPollen dataset will be made publicly available
after acceptance of the paper**, to support full reproducibility of the experiments. Until then, the
**[live demo](https://richardesp-pollenformer-uco-demo.hf.space/)** lets you exercise the trained
model directly in the browser.

---

## 📚 Citation

> The author list and venue are withheld during peer review and will be completed upon acceptance.

```bibtex
@unpublished{pollenformeruco2026,
  title  = {Vision Transformers for Pollen Grains Classification:
            A Systematic Backbone Benchmark on UCOPollen},
  author = {Author names withheld for peer review},
  year   = {2026},
  note   = {Manuscript under review. Full author list and venue added upon acceptance.}
}
```

---

## 👥 Authors

Author and affiliation details are **withheld during peer review** and will be added upon acceptance.

---

## 📄 License

Released under the **MIT License**. A `LICENSE` file will accompany the code release.
