# MNIST TCAV Consolidated Report

## 1. Scope

This report consolidates results from two experiment tracks:

- Single-concept baseline: outputs/run_tcav/report/tcav_report.md and summary.json
- Multi-concept (PyTorch CAV): outputs/stroke_concepts_pytorch/report/stroke_concepts_report_pytorch.md and stroke_concepts_summary_pytorch.json

## 2. Executive Summary

- All tracks use a classifier with the same test accuracy: 0.9876, indicating stable predictive performance.
- X-shape is not supported as a positive explanatory concept for digit 8 in either single- or multi-concept analysis.
- Under larger sampling settings (PyTorch CAV: runs=20, concept_size=400, eval_size=512):
  - x_shape and loop are significantly negative for class 8 (mean < 0.5 and one-sided p(mean < 0.5) < 0.05).
  - slash, vertical, horizontal, and backslash are not significant for class 8.
- Several directional concepts show higher TCAV for class 3 than class 8 (notably horizontal/backslash/vertical), suggesting concept definitions may align more with class-3 evidence in this setup.

## 3. Key Results

### 3.1 Single-concept X-shape baseline

- class-8 mean = 0.0391, 95% CI = [-0.0183, 0.0965]
- class-3 mean = 0.0187, 95% CI = [-0.0172, 0.0547]
- gap (8 - 3) = 0.0203

Interpretation:

- Both means are far below 0.5; confidence intervals overlap zero strongly.
- Evidence is weak that X-shape increases class-8 logits.

### 3.2 Multi-concept PyTorch CAV (runs=20)

Ordered by class-8 mean (as in report):

- slash: 0.5017 (not significant)
- vertical: 0.4895 (not significant)
- horizontal: 0.3729 (not significant)
- backslash: 0.3375 (not significant)
- x_shape: 0.1451 (significant negative, p_lt about 2.88e-7)
- loop: 0.1080 (significant negative, p_lt about 2.10e-14)

Interpretation:

- Under stronger sampling, only x_shape and loop remain significant negatives.

## 4. Complementary Findings: Baseline vs PyTorch CAV

Consistent findings:

- x_shape is not a positive concept for class 8.
- In multi-concept PyTorch CAV, both x_shape and loop are significantly negative for class 8.

Complementary information:

- The single-concept baseline quantifies overall X-shape effect size (means, CIs, and 8-3 gap).
- Multi-concept PyTorch CAV provides cross-concept ranking and significance, and shows several directional concepts are stronger for class 3 than class 8 in this setup.

## 5. Statistical and Methodological Notes

- Significance is defined with one-sided tests relative to 0.5:
  - Positive dependency: mean > 0.5 and p(mean > 0.5) < 0.05
  - Negative dependency: mean < 0.5 and p(mean < 0.5) < 0.05
- Conclusions from larger runs and larger concept/eval sets are generally more reliable; therefore the PyTorch-CAV track should be treated as the primary evidence for multi-concept claims.

## 6. Final Interpretation

- In this project setting, digit-8 recognition is not driven by the X-shape concept.
- X-shape behaves more like a counter-evidence concept for class 8.
- The current concept library likely under-represents shape primitives that truly characterize digit 8 (for example, loop closure structure), so explanatory coverage is incomplete.

## 7. Recommended Next Steps

- Expand concept taxonomy with digit-8-oriented morphology (upper/lower loop, waist pinch, dual-loop closure, intersection variants).
- Build concept sets from real curated samples in addition to synthetic generation.
- Repeat TCAV at multiple bottleneck layers.
- Run multiple full-seed repeats and report confidence intervals on top-level conclusions.
- Add comparative targets (for example 0, 6, 9) to isolate loop-related concept boundaries.
