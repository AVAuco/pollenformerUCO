---
title: PollenFormerUCO Demo
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Interactive Streamlit demo for PollenFormerUCO
---

# PollenFormerUCO · Pollen Image Classifier

Interactive demo of **PollenFormerUCO**, accompanying the paper
[_Vision Transformers for pollen grains classification: A systematic backbone benchmark on UCOPollen_](https://doi.org/10.1016/j.atmosenv.2026.122455)
(**Atmospheric Environment**, 389, 122455, 2027).

> 🛠️ This Space is deployed automatically from
> [github.com/AVAuco/pollenformerUCO](https://github.com/AVAuco/pollenformerUCO), the single
> source of truth for the code. Please open issues and pull requests there.

PollenFormerUCO is a **DINOv2-ViT Base** backbone fine-tuned with a **hybrid Cross-Entropy +
Triplet loss** that projects the CLS token into a **128-D embedding**. Predictions come from
**cosine k-nearest-neighbour** search over a gallery of training embeddings, and the app overlays
the model's **attention map** to show which regions drove the decision.

It recognises **18 classes**: 17 pollen taxa plus a negative `NoPollen` class. On the UCOPollen
test split, cosine 1-NN reaches **98.1%** accuracy and **97.5%** macro-F1.

Upload a JPG, PNG, WEBP or BMP image (up to 10 MiB) or pick a sample, click **Predict**, and
inspect the predicted taxon, its nearest neighbours and the attention overlay. Set the Space
variable `UI_MODE=HARD` for the expert interface.
