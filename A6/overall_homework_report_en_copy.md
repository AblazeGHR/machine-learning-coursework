# A6: TCAV Analysis on MNIST

Name: GUO Haorong
ID: 12511217

## Abstract

Main conclusions: the trained MNIST classifier achieves stable performance on the test set (test accuracy = 0.9876). The `x_shape` concept does not act as a positive explanatory concept for digit 8. In more comprehensive multi-concept (PyTorch CAV) experiments, `x_shape` and `loop` show significant negative dependency with digit 8 (mean < 0.5, one-sided p < 0.05). The other directional concepts (`slash`, `vertical`, `horizontal`, `backslash`) do not show significant positive effects for digit 8. This document provides methods, results, possible mechanisms, and proposed follow-up experiments.

## 1. Background and Objectives

The interpretability method TCAV (Testing with Concept Activation Vectors) is used to assess how semantic concepts affect model outputs. The objectives of this project are:

- Train a simple MNIST CNN as the target model;
- Evaluate whether hand-crafted / programmatically generated stroke concepts such as X-shape increase or decrease the classifier's evidence for digit 8;
- Perform a horizontal comparison across multiple concepts and determine significance to form preliminary explanations for the model's decision behavior.

## 2. Methods and Experimental Setup

- Dataset: MNIST (the code uses torchvision datasets).
- Model architecture: a simple two-layer convolutional neural network.
- Main workflow:
  - Train the model and generate the single-concept X-shape TCAV evaluation in `run_tcav.py`;
  - Use `run_stroke_concepts_tcav_torch_cav.py` to train PyTorch-based CAVs and evaluate TCAV scores for six stroke concepts (`x_shape`, `vertical`, `horizontal`, `slash`, `backslash`, `loop`).
- Key parameters (PyTorch CAV experiments as primary): `runs=20`, `concept_size=400`, `eval_size=512` (representing higher sampling); the single-concept baseline used `runs=10`, `concept_size=300`, `eval_size=256`.
- Significance decision: one-sided test relative to 0.5 (i.e., mean > 0.5 and p(mean > 0.5) < 0.05 is considered positive dependency; mean < 0.5 and p(mean < 0.5) < 0.05 is considered negative dependency).

## 3. Results

### 3.1 Model performance

- Test accuracy: 0.9876 (used to judge whether the model quality is sufficient to support subsequent interpretability analysis).

### 3.2 Single-concept baseline (X-shape)

(`CI` denotes Confidence Interval, representing the uncertainty range of the estimate, e.g., 95% CI.)
- class-8 TCAV mean = 0.0391, 95% CI = [-0.0183, 0.0965]
- class-3 TCAV mean = 0.0187, 95% CI = [-0.0172, 0.0547]
- gap (8 - 3) = 0.0203

Explanation: the means are far below 0.5 and the confidence intervals include 0, indicating no evidence in the single-concept setting that X-shape increases the class-8 logit.

### 3.3 Multi-concept (PyTorch CAV, runs=20) comparison

Ordered by class-8 mean (as in the original report):
(`p_lt` denotes the left-sided one-sided p-value testing mean < 0.5; smaller values indicate stronger evidence for mean < 0.5.)
- slash: 0.5017 (not significant)
- vertical: 0.4895 (not significant)
- horizontal: 0.3729 (not significant)
- backslash: 0.3375 (not significant)
- x_shape: 0.1451 (significant negative, p_lt ≈ 2.88e-7)
- loop: 0.1080 (significant negative, p_lt ≈ 2.10e-14)

Explanation: with higher sampling, only `x_shape` and `loop` show robust negative dependency; the other directional concepts do not show significant positive effects for digit 8.

![TCAV gap (class8 - class3)](outputs/stroke_concepts_pytorch/figures/stroke_concepts_gap_8_minus_3_pytorch.png)

- Figure 1: TCAV gap for concepts on digit 8 (class8 - class3). Bar height is the gap; negative values indicate the concept supports digit 8 less than digit 3, useful for identifying concept class-specificity.

![TCAV scores for class 8 (means & error bars)](outputs/stroke_concepts_pytorch/figures/stroke_concepts_tcav8_pytorch.png)

- Figure 2: TCAV scores for digit 8 (means and error bars). Bars indicate means; error bars indicate inter-sample variation; the dashed line at 0.5 is the reference threshold for the one-sided dependency test.

Explanation: Figures 1 and 2 are complementary. Figure 2 assesses the absolute effect size and uncertainty for each concept on digit 8; Figure 1 shows the relative bias between digit 8 and digit 3, helping to determine whether a concept favors 3 or 8.

## 4. Analysis and Discussion

### 4.1 Evidence strength for main conclusions

- The single-concept baseline provides overall effect size estimates (mean, CI) and shows `x_shape` has negligible effect;
- PyTorch CAV (higher runs) provides more stable significance judgments for multi-concept comparison, supporting `x_shape` and `loop` as inhibitory concepts for digit 8 under current concept constructions;
- Since concept samples are programmatically generated, they may not cover key morphological variants present in real handwriting, which affects interpretability depth.

### 4.2 Possible mechanistic hypotheses

- One possible explanation: generated `x_shape` samples may visually resemble digit 3, or trigger intermediate features associated with 3, leading to lower scores for 8;
- The `loop` concept (representing closed or ring-like structures) may lack the positive feature variants that the classifier relies on for recognizing 8, or generated samples differ from real loop shapes;
- Another possibility is that local stroke directions (horizontal/vertical/slash/backslash) are treated by the model primarily as cues distinguishing 3 vs 8 rather than as direct evidence for closed-loop structures of 8.

> These are hypotheses based on current observations and require additional experiments for validation (see Section 7 recommendations).

### 4.3 Limitations and potential issues

- Concept sets are mainly programmatically generated; representativeness for real handwriting may be limited;
- The 0.5 threshold and one-sided test depend on concept generation and evaluation sample representativeness;
- TCAV was not repeated across multiple bottleneck layers; conclusions may depend on the chosen bottleneck layer;
- Some experiments (single-concept baseline) used a small number of runs, resulting in less stable variance estimates; prioritize high-run experiments for conclusions.

## 5. Conclusion

Under the current project setup and concept definitions, we cannot support the claim that `x_shape` is a positive explanatory concept for digit 8. Instead, `x_shape` and `loop` show significant negative dependency with digit 8 in multi-concept, high-sampling PyTorch CAV experiments. Researchers using TCAV for interpretability claims should carefully define concept representativeness and use sufficient sampling to reduce spurious conclusions.

## 6. Follow-up recommendations (experimental design points)

- Expand the concept library with 8-oriented morphologies such as upper/lower loops, dual-loop closure, and waist-pinching shapes;
- Construct concept sets from real MNIST samples by manual curation to improve concept representativeness;
- Repeat TCAV at multiple bottleneck layers to compare concept responses across layers;
- Repeat the full experiment pipeline multiple times with fixed seeds (at least 3) and report aggregated means and confidence intervals;
- If the focus is on closed-loop structures, include digits `0`, `6`, and `9` as comparison targets to test loop-concept specificity.
