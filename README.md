# Vision Transformer with Edge-Weighted Patch Embedding

This repository contains a custom implementation of the Vision Transformer (ViT) architecture, introducing a novel **Edge-Weighted Patch Embedding** mechanism. The project compares this custom approach against standard pre-trained ViT models (using JAX/Flax on TPU) and a ResNet50 baseline on the CIFAR-10 dataset.

## 🚀 Novelty: Edge-Weighted Patches

The core innovation in this project is the integration of edge detection into the transformer's embedding layer. Instead of treating all image patches equally during projection, this model re-weights them based on their structural content.

* **Sobel Filter Integration:** A custom Sobel filter is applied to the input image to calculate gradient magnitudes and detect edges.
* **Dynamic Weighting:** The model calculates the mean edge intensity for each patch. These intensities are used as weights to scale the patch embeddings element-wise before they enter the Transformer Encoder.
* **Hypothesis:** This inductive bias helps the model focus on structurally significant areas of the image (shapes and contours) earlier in the processing pipeline.

## 📂 Project Overview

The notebook `visionTransformer.ipynb` covers three distinct experimental phases:

### 1. Custom ViT Implementation (PyTorch)
A modular ViT built from scratch using `torch` and `einops`, featuring:
* **Custom `PatchEmbedding` Class:** Modified to accept and apply edge-based weights to the projected tokens.
* **Transformer Components:** Manual implementation of Multi-Head Self Attention (MSA), MLP blocks, and Residual connections.
* **Hybrid Model:** Integration with `timm` (PyTorch Image Models) to leverage pre-trained backbones while injecting the custom embedding logic.

### 2. TPU Training (JAX/Flax)
Implementation of a standard ViT training pipeline optimized for Google Colab TPUs:
* Utilizes the `vision_transformer` library (JAX/Flax) from Google Research.
* Includes code for model configuration, checkpoint loading, and high-performance training loops on TPU cores.

### 3. Baseline Comparison (ResNet50)
A standard ResNet50 model implemented in TensorFlow/Keras is trained on CIFAR-10 to serve as a performance baseline for the transformer experiments.

## 🛠️ Requirements

* Python 3.x
* PyTorch
* Torchvision
* Einops
* Timm
* JAX / Flax (for TPU experiments)
* TensorFlow (for ResNet baseline)
* Matplotlib / Numpy

## 💻 Usage

The entire project logic is contained within the Jupyter Notebook.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/shrivi34/VisionTransformer_ViT.git](https://github.com/shrivi34/VisionTransformer_ViT.git)
    cd VisionTransformer_ViT
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision einops timm jax tensorflow
    ```

3.  **Run the Notebook:**
    Open `visionTransformer.ipynb` in Google Colab or Jupyter Lab.
    * *Note: To run the JAX/Flax section, ensure you are using a TPU runtime.*

## 📊 Dataset

The model is trained and evaluated on the **CIFAR-10** dataset, consisting of 60,000 32x32 color images in 10 classes.

## 📚 References

* **ViT Paper:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* **Google ViT Repository:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
