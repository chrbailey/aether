# AETHER Benchmark Comparison Report

Generated: 2026-01-30 18:42 UTC

## Overview

This report compares AETHER's adaptive threshold governance against static thresholds across multiple process mining datasets. The key metric is **MCC (Matthews Correlation Coefficient)**, which balances true/false positives and negatives for the review gate decision.

## Summary Results

| Dataset | Domain | Cases | Accuracy | ECE | Static MCC | AETHER MCC | Improvement |
|---------|--------|------:|:--------:|:---:|:----------:|:----------:|:-----------:|
| Sepsis | Healthcare | 210 | 55.2% | 0.1335 | 0.7077 | 0.7241 | +2.3% |
| BPI 2019 | Finance | 500 | 88.8% | 0.2486 | 0.4599 | 0.4626 | +0.6% |
| BPIC 2012 | Finance | 500 | 69.2% | 0.1482 | 0.2654 | 0.2665 | +0.4% |
| NetSuite 2025 | Finance | 274 | 40.9% | N/A | 0.1693 | 0.1637 | -3.3% |
| SAP BSP669 | Enterprise | 767 | 37.5% | N/A | 0.2514 | 0.1910 | -24.0% |
| SAP Workflow | Enterprise | 2896 | 95.4% | N/A | 0.1585 | 0.2081 | +31.3% |
| Wearable Tracker | Retail | 218 | 75.7% | N/A | 0.2935 | 0.3458 | +17.8% |
| Judicial | Legal | 5 | 40.0% | N/A | 0.0000 | 0.0000 | 0.0% |
| BPI 2018 Agriculture | Government | 2000 | 75.8% | N/A | 0.3479 | 0.3262 | -6.2% |
| Road Traffic Fine | Government | 30074 | 55.6% | N/A | -0.0077 | 0.0128 | +266.2% |

## Aggregate Statistics

- **Total Datasets:** 10 / 8
- **Total Cases Evaluated:** 37,444
- **Average Model Accuracy:** 63.4%
- **Average Static MCC:** 0.2646
- **Average AETHER MCC:** 0.2701
- **Average MCC Improvement:** +28.5%

## Key Findings

1. **Best Domain for Adaptive Thresholds:** Government (avg +130.0% MCC improvement)
2. **Highest Individual Improvement:** Road Traffic Fine (+266.2% MCC improvement)
3. **Regressions Observed:** SAP BSP669, NetSuite 2025, BPI 2018 Agriculture - static thresholds outperformed adaptive in these cases
4. **High ECE Datasets:** Sepsis (ECE=0.133), BPI 2019 (ECE=0.249), BPIC 2012 (ECE=0.148) - may benefit from additional calibration
5. **Average Review Burden Change:** +5.6% (reduction with AETHER)

## Domain Analysis

### Healthcare

**Sepsis** (Hospital sepsis case management)
- Cases: 210, Accuracy: 55.2%
- Static MCC: 0.7077, AETHER MCC: 0.7241 (improvement: 2.3%)
- Review burden: 53.3% (static) vs 52.4% (AETHER)

**Domain Summary:** 1 dataset(s), Avg Static MCC: 0.7077, Avg AETHER MCC: 0.7241, Avg Improvement: +2.3%

### Finance

**BPI 2019** (Procurement purchase orders)
- Cases: 500, Accuracy: 88.8%
- Static MCC: 0.4599, AETHER MCC: 0.4626 (improvement: 0.6%)
- Review burden: 22.7% (static) vs 23.4% (AETHER)

**BPIC 2012** (Loan application processing)
- Cases: 500, Accuracy: 69.2%
- Static MCC: 0.2654, AETHER MCC: 0.2665 (improvement: 0.4%)
- Review burden: 30.4% (static) vs 27.6% (AETHER)

**NetSuite 2025** (Financial transactions)
- Cases: 274, Accuracy: 40.9%
- Static MCC: 0.1693, AETHER MCC: 0.1637 (regression: 3.3%)
- Review burden: 19.7% (static) vs 8.0% (AETHER)

**Domain Summary:** 3 dataset(s), Avg Static MCC: 0.2982, Avg AETHER MCC: 0.2976, Avg Improvement: -0.8%

### Enterprise

**SAP Workflow** (Synthetic O2C/P2P workflows)
- Cases: 2896, Accuracy: 95.4%
- Static MCC: 0.1585, AETHER MCC: 0.2081 (improvement: 31.3%)
- Review burden: 11.4% (static) vs 8.3% (AETHER)

**SAP BSP669** (Enterprise ERP transactions)
- Cases: 767, Accuracy: 37.5%
- Static MCC: 0.2514, AETHER MCC: 0.1910 (regression: 24.0%)
- Review burden: 53.3% (static) vs 25.3% (AETHER)

**Domain Summary:** 2 dataset(s), Avg Static MCC: 0.2050, Avg AETHER MCC: 0.1996, Avg Improvement: +3.6%

### Retail

**Wearable Tracker** (Customer journey O2C)
- Cases: 218, Accuracy: 75.7%
- Static MCC: 0.2935, AETHER MCC: 0.3458 (improvement: 17.8%)
- Review burden: 17.9% (static) vs 15.6% (AETHER)

**Domain Summary:** 1 dataset(s), Avg Static MCC: 0.2935, Avg AETHER MCC: 0.3458, Avg Improvement: +17.8%

### Legal

**Judicial** (Court case proceedings (novel domain))
- Cases: 5, Accuracy: 40.0%
- Static MCC: 0.0000, AETHER MCC: 0.0000 (regression: 0.0%)
- Review burden: 100.0% (static) vs 100.0% (AETHER)

**Domain Summary:** 1 dataset(s), Avg Static MCC: 0.0000, Avg AETHER MCC: 0.0000, Avg Improvement: +0.0%

### Government

**Road Traffic Fine** (Italian traffic fine management (150K cases))
- Cases: 30074, Accuracy: 55.6%
- Static MCC: -0.0077, AETHER MCC: 0.0128 (improvement: 266.2%)
- Review burden: 7.5% (static) vs 2.9% (AETHER)

**BPI 2018 Agriculture** (German agricultural subsidy applications)
- Cases: 2000, Accuracy: 75.8%
- Static MCC: 0.3479, AETHER MCC: 0.3262 (regression: 6.2%)
- Review burden: 30.1% (static) vs 27.1% (AETHER)

**Domain Summary:** 2 dataset(s), Avg Static MCC: 0.1701, Avg AETHER MCC: 0.1695, Avg Improvement: +130.0%

## Methodology

### Evaluation Protocol

1. **Ground Truth:** A case needs review if the model's next-activity prediction is wrong
2. **Static Baseline:** Fixed threshold of 0.55 for the reviewGateAutoPass gate
3. **AETHER Adaptive:** Dynamic threshold using v2 bidirectional formula:
   - `effective_threshold = base * mode_factor * uncertainty_factor * calibration_factor`
4. **Metrics:** MCC, F1, precision, recall, review burden (% of cases flagged)

### Governance Modes

| Mode | Factor | Description |
|------|--------|-------------|
| Flexible | 1.0 | Lower thresholds, fewer reviews |
| Standard | 1.1 | Balanced (default) |
| Strict | 1.2 | Higher thresholds, more reviews |

Results shown use **standard** mode unless otherwise noted.
