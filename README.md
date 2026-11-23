# European AI Championship 2026

This repository contains our work from the **European AI Championship 2026**, a 24 hour AI hackathon held in Paris on 21–22 November 2026.

The competition was organised by researchers from **CISPA**, **Fraunhofer AISEC**, and **SprintML**, with a focus on building secure, private, robust, interpretable, and trustworthy machine learning systems.

Participants tackled three technical challenges covering image watermark localisation, language model identification, and machine unlearning.

## Task 1: Image Watermark Localisation

**Objective:** Identify which regions of an image contain an embedded digital watermark.

Digital watermarks are hidden signals embedded into images to establish their origin, authenticity, or ownership. Unlike visible logos, these watermarks can be difficult to detect by simply looking at an image.

The challenge was to develop a system capable of detecting and precisely locating watermarked regions, even when only part of an image contains a watermark.

### Challenge Details

- **Input:** Images with a resolution of 512 × 512 pixels.
- **Output:** A binary segmentation mask of size 128 × 128.
- **Granularity:** Each predicted cell represents a 4 × 4 pixel region.
- **Classification:** Assign 1 if at least 50% of a cell is watermarked, otherwise 0.
- **Watermarking methods:** PixelSeal, TrustMark, and StegaStamp.

The dataset contained **1,530 images**, equally distributed across the three watermarking methods:

- **Training:** 510 images, with ground truth masks, watermarking methods, and embedded messages.
- **Testing:** 1,020 images, with watermarking methods and embedded messages but no ground truth masks.

### Evaluation

Performance was measured using **mean Intersection over Union (mIoU)**, comparing predicted watermark regions against the actual watermarked areas.

IoU was calculated separately for watermarked and clean regions, averaged per image, and then averaged across the evaluation dataset.

**Goal:** Accurately identify watermarked pixels while minimising false positives and missed watermark regions.

## Task 2: Model Equality Testing

**Objective:** Determine whether different language model API endpoints are running the same underlying AI model.

Two AI services can produce very similar responses without necessarily using identical models. Conversely, the same model can produce different outputs because of changes to its system prompt, watermark, or text generation strategy.

The challenge was to identify the underlying models using only the text returned by their API endpoints.

### Challenge Details

Participants received access to **33 language model endpoints** and had to solve two related problems:

1. **Model family identification:** Determine which endpoints share the same base model.
2. **Model identity verification:** Determine which endpoints run the same model, accounting for differences in deployment configuration.

Changes to the base model, quantisation, or LoRA adapters were considered different model identities. Changes to system prompts, watermarks, or decoding strategies did not change model identity.

The challenge required designing informative prompts, analysing response patterns, and identifying subtle behavioural differences between models.

### Constraints

- Maximum of 100 responses per endpoint during the competition.
- Maximum of 1,024 input tokens per prompt.
- Maximum of 256 generated tokens per response.
- Up to 4 concurrent API requests.
- No access to model weights or internal configurations.

### Evaluation

Submissions were evaluated using the **Matthews Correlation Coefficient (MCC)** across 528 endpoint pairs.

The final score was weighted as follows:

- **10%:** Correct identification of model families.
- **90%:** Correct identification of model identities within families.

**Submission:** A CSV file assigning a family and model identifier to each endpoint.

**Goal:** Develop an accurate and query efficient method for identifying AI models using only black box API access.

## Task 3: Machine Unlearning

**Objective:** Remove the influence of specific training images from an existing neural network without significantly damaging its overall performance.

Machine learning models can memorise information from their training data. This creates challenges when particular data must be removed, for example because of privacy requirements, copyright concerns, or data deletion requests.

Rather than retraining an entire model from scratch, machine unlearning aims to modify an existing model so that it behaves as though certain training examples had never been included.

### Challenge Details

Participants received:

- **Model:** A pretrained Vision Transformer (ViT B/16).
- **Dataset:** A custom image classification dataset containing 100 classes.
- **Forget set:** 3,500 training images that the model must forget.
- **Available information:** Original model weights and forget images without labels.

The task was to modify the original model so that its behaviour on the forget set resembles a reference model trained from scratch without those images.

Simply causing the model to misclassify forgotten images was not sufficient. Successful unlearning required matching the reference model's prediction distributions and loss statistics, including on altered versions of the forgotten images.

### Constraints

- No additional labelled training data was provided.
- The original model architecture had to remain unchanged.
- The submitted weights had to match the original architecture exactly.
- Submitted parameters had to be float32.
- Maximum model file size was 450 MB.

### Evaluation

The final score combined two components:

**1. Forget Quality**

Measured how closely the submitted model matched the reference model on forgotten images and hidden transformations of those images.

It considered:

- Similarity of cross entropy loss distributions.
- Similarity of predicted class probability distributions using KL divergence.

**2. Retain Quality**

Measured how effectively the modified model preserved its original capabilities.

It considered:

- Classification accuracy on unseen control images.
- Similarity to the original model's prediction distributions.

The overall score was calculated as:

**Final Score = Forget Score × Retain Score**

An additional requirement imposed a minimum of **70% classification accuracy** on a hidden validation dataset. Models falling below this threshold received a score of zero.

**Submission:** A PyTorch state dictionary containing the modified model weights.

**Goal:** Achieve effective forgetting while maintaining the original model's generalisation and classification performance.

## Competition Infrastructure

Participants were provided access to GPU computing resources through the **Jülich Supercomputing Centre (JSC)**.

The available compute nodes included:

- 4 NVIDIA A100 GPUs per node, each with 40 GB of memory.
- Dual AMD EPYC 7742 processors.
- 512 GB of system memory.

Solutions were evaluated through competition leaderboards, with public scores available during the event and hidden evaluation data used for final rankings.

## Resources

- [SprintML Hackathon Repository](https://huggingface.co/datasets/SprintML/hackathon)
- [Machine Unlearning Dataset](https://huggingface.co/datasets/SprintML/unlearning)
- [Towards Unbounded Machine Unlearning (NeurIPS 2023)](https://arxiv.org/abs/2302.09880)

---

**European AI Championship Paris 2026** | 21–22 November 2026
