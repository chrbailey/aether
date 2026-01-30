"""
Generate comparison report from AETHER benchmark results.

Compiles results from all 10 benchmark datasets and generates:
1. Markdown report at docs/BENCHMARK_COMPARISON.md
2. JSON summary at data/benchmarks/aggregate_results.json

Datasets:
- sepsis.json (Healthcare)
- bpi2019.json (Finance/Procurement)
- bpic2012.json (Finance/Loan Application)
- wearable_tracker.json (Retail O2C)
- sap_workflow.json (Enterprise O2C/P2P)
- sap_bsp669.json (Enterprise ERP)
- netsuite_2025.json (Finance/Transactions)
- judicial.json (Legal - Novel Domain)
- road_traffic.json (Government/Traffic Fines)
- bpi2018.json (Government/Agriculture Subsidies)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
BENCHMARKS_DIR = AETHER_ROOT / "data" / "benchmarks"
DOCS_DIR = AETHER_ROOT / "docs"

# Dataset configuration with domain mappings
DATASETS = {
    "sepsis": {
        "file": "sepsis.json",
        "label": "Sepsis",
        "domain": "Healthcare",
        "description": "Hospital sepsis case management",
    },
    "bpi2019": {
        "file": "bpi2019.json",
        "label": "BPI 2019",
        "domain": "Finance",
        "description": "Procurement purchase orders",
    },
    "bpic2012": {
        "file": "bpic2012.json",
        "label": "BPIC 2012",
        "domain": "Finance",
        "description": "Loan application processing",
    },
    "wearable_tracker": {
        "file": "wearable_tracker.json",
        "label": "Wearable Tracker",
        "domain": "Retail",
        "description": "Customer journey O2C",
    },
    "sap_workflow": {
        "file": "sap_workflow.json",
        "label": "SAP Workflow",
        "domain": "Enterprise",
        "description": "Synthetic O2C/P2P workflows",
    },
    "sap_bsp669": {
        "file": "sap_bsp669.json",
        "label": "SAP BSP669",
        "domain": "Enterprise",
        "description": "Enterprise ERP transactions",
    },
    "netsuite_2025": {
        "file": "netsuite_2025.json",
        "label": "NetSuite 2025",
        "domain": "Finance",
        "description": "Financial transactions",
    },
    "judicial": {
        "file": "judicial.json",
        "label": "Judicial",
        "domain": "Legal",
        "description": "Court case proceedings (novel domain)",
    },
    "road_traffic": {
        "file": "road_traffic.json",
        "label": "Road Traffic Fine",
        "domain": "Government",
        "description": "Italian traffic fine management (150K cases)",
    },
    "bpi2018": {
        "file": "bpi2018.json",
        "label": "BPI 2018 Agriculture",
        "domain": "Government",
        "description": "German agricultural subsidy applications",
    },
}

DOMAIN_ORDER = ["Healthcare", "Finance", "Enterprise", "Retail", "Legal", "Government"]


def load_benchmark(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a benchmark JSON file, return None if not found or invalid."""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: Failed to load {file_path.name}: {e}")
        return None


def extract_metrics(data: dict[str, Any], mode: str = "standard") -> dict[str, Any]:
    """Extract key metrics from benchmark data for a given mode."""
    dataset = data.get("dataset", {})
    per_mode = data.get("per_mode", {})
    mode_data = per_mode.get(mode, {})

    static = mode_data.get("static", {})
    aether = mode_data.get("aether", {})

    static_mcc = static.get("mcc", 0)
    aether_mcc = aether.get("mcc", 0)

    if static_mcc != 0:
        improvement = ((aether_mcc - static_mcc) / abs(static_mcc)) * 100
    elif aether_mcc != 0:
        improvement = 100.0
    else:
        improvement = 0.0

    # Get ECE from model calibration
    model = data.get("model", {})
    calibration = model.get("calibration", {})
    ece = calibration.get("ece", 0)

    return {
        "cases": dataset.get("total_cases", dataset.get("evaluated", 0)),
        "accuracy": dataset.get("accuracy", 0),
        "ece": ece,
        "static_mcc": static_mcc,
        "aether_mcc": aether_mcc,
        "improvement": improvement,
        "static_f1": static.get("f1", 0),
        "aether_f1": aether.get("f1", 0),
        "static_burden": static.get("burden", 0),
        "aether_burden": aether.get("burden", 0),
        "review_rate": dataset.get("review_rate", 0),
    }


def generate_summary_table(results: dict[str, dict]) -> str:
    """Generate markdown summary table."""
    lines = [
        "| Dataset | Domain | Cases | Accuracy | ECE | Static MCC | AETHER MCC | Improvement |",
        "|---------|--------|------:|:--------:|:---:|:----------:|:----------:|:-----------:|",
    ]

    # Sort by domain order, then by dataset name
    sorted_keys = sorted(
        results.keys(),
        key=lambda k: (
            DOMAIN_ORDER.index(DATASETS[k]["domain"])
            if DATASETS[k]["domain"] in DOMAIN_ORDER
            else 99,
            k,
        ),
    )

    for key in sorted_keys:
        cfg = DATASETS[key]
        metrics = results[key]

        accuracy_str = f"{metrics['accuracy']:.1%}"
        ece_str = f"{metrics['ece']:.4f}" if metrics["ece"] else "N/A"
        improvement_str = f"+{metrics['improvement']:.1f}%" if metrics["improvement"] > 0 else f"{metrics['improvement']:.1f}%"

        lines.append(
            f"| {cfg['label']} | {cfg['domain']} | {metrics['cases']} | "
            f"{accuracy_str} | {ece_str} | {metrics['static_mcc']:.4f} | "
            f"{metrics['aether_mcc']:.4f} | {improvement_str} |"
        )

    return "\n".join(lines)


def generate_domain_analysis(results: dict[str, dict]) -> str:
    """Generate per-domain analysis section."""
    sections = []

    for domain in DOMAIN_ORDER:
        domain_datasets = [
            (key, results[key])
            for key in results
            if DATASETS[key]["domain"] == domain
        ]

        if not domain_datasets:
            continue

        section_lines = [f"### {domain}"]

        for key, metrics in domain_datasets:
            cfg = DATASETS[key]
            improvement_dir = "improvement" if metrics["improvement"] > 0 else "regression"
            section_lines.append(
                f"\n**{cfg['label']}** ({cfg['description']})\n"
                f"- Cases: {metrics['cases']}, Accuracy: {metrics['accuracy']:.1%}\n"
                f"- Static MCC: {metrics['static_mcc']:.4f}, AETHER MCC: {metrics['aether_mcc']:.4f} "
                f"({improvement_dir}: {abs(metrics['improvement']):.1f}%)\n"
                f"- Review burden: {metrics['static_burden']:.1%} (static) vs {metrics['aether_burden']:.1%} (AETHER)"
            )

        # Domain summary
        avg_improvement = sum(m["improvement"] for _, m in domain_datasets) / len(domain_datasets)
        avg_static_mcc = sum(m["static_mcc"] for _, m in domain_datasets) / len(domain_datasets)
        avg_aether_mcc = sum(m["aether_mcc"] for _, m in domain_datasets) / len(domain_datasets)

        section_lines.append(
            f"\n**Domain Summary:** {len(domain_datasets)} dataset(s), "
            f"Avg Static MCC: {avg_static_mcc:.4f}, Avg AETHER MCC: {avg_aether_mcc:.4f}, "
            f"Avg Improvement: {avg_improvement:+.1f}%"
        )

        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


def generate_key_findings(results: dict[str, dict]) -> str:
    """Generate key findings section."""
    findings = []

    # Sort by improvement
    by_improvement = sorted(results.items(), key=lambda x: x[1]["improvement"], reverse=True)

    # Best performing domain
    domain_improvements = {}
    for key, metrics in results.items():
        domain = DATASETS[key]["domain"]
        if domain not in domain_improvements:
            domain_improvements[domain] = []
        domain_improvements[domain].append(metrics["improvement"])

    domain_avg = {
        d: sum(imps) / len(imps) for d, imps in domain_improvements.items()
    }
    best_domain = max(domain_avg.items(), key=lambda x: x[1])
    worst_domain = min(domain_avg.items(), key=lambda x: x[1])

    findings.append(
        f"1. **Best Domain for Adaptive Thresholds:** {best_domain[0]} "
        f"(avg +{best_domain[1]:.1f}% MCC improvement)"
    )

    if by_improvement:
        best_key, best_metrics = by_improvement[0]
        findings.append(
            f"2. **Highest Individual Improvement:** {DATASETS[best_key]['label']} "
            f"(+{best_metrics['improvement']:.1f}% MCC improvement)"
        )

    # Datasets with negative improvement
    negative = [(k, m) for k, m in results.items() if m["improvement"] < 0]
    if negative:
        neg_names = ", ".join(DATASETS[k]["label"] for k, _ in negative)
        findings.append(
            f"3. **Regressions Observed:** {neg_names} - "
            f"static thresholds outperformed adaptive in these cases"
        )
    else:
        findings.append(
            "3. **No Regressions:** AETHER adaptive thresholds improved or matched "
            "static thresholds across all datasets"
        )

    # Calibration correlation
    high_ece = [(k, m) for k, m in results.items() if m["ece"] > 0.1]
    if high_ece:
        ece_names = ", ".join(f"{DATASETS[k]['label']} (ECE={m['ece']:.3f})" for k, m in high_ece)
        findings.append(
            f"4. **High ECE Datasets:** {ece_names} - "
            f"may benefit from additional calibration"
        )

    # Review burden reduction
    total_burden_reduction = sum(
        m["static_burden"] - m["aether_burden"] for m in results.values()
    ) / len(results) * 100
    findings.append(
        f"5. **Average Review Burden Change:** {total_burden_reduction:+.1f}% "
        f"({'reduction' if total_burden_reduction > 0 else 'increase'} with AETHER)"
    )

    return "\n".join(findings)


def generate_aggregate_stats(results: dict[str, dict]) -> dict[str, Any]:
    """Generate aggregate statistics."""
    total_cases = sum(m["cases"] for m in results.values())
    avg_accuracy = sum(m["accuracy"] for m in results.values()) / len(results)
    avg_static_mcc = sum(m["static_mcc"] for m in results.values()) / len(results)
    avg_aether_mcc = sum(m["aether_mcc"] for m in results.values()) / len(results)
    avg_improvement = sum(m["improvement"] for m in results.values()) / len(results)

    # Per-domain aggregates
    domain_stats = {}
    for domain in DOMAIN_ORDER:
        domain_results = [
            m for k, m in results.items() if DATASETS[k]["domain"] == domain
        ]
        if domain_results:
            domain_stats[domain] = {
                "n_datasets": len(domain_results),
                "total_cases": sum(m["cases"] for m in domain_results),
                "avg_accuracy": sum(m["accuracy"] for m in domain_results) / len(domain_results),
                "avg_static_mcc": sum(m["static_mcc"] for m in domain_results) / len(domain_results),
                "avg_aether_mcc": sum(m["aether_mcc"] for m in domain_results) / len(domain_results),
                "avg_improvement": sum(m["improvement"] for m in domain_results) / len(domain_results),
            }

    return {
        "total_datasets": len(results),
        "total_cases": total_cases,
        "avg_accuracy": round(avg_accuracy, 4),
        "avg_static_mcc": round(avg_static_mcc, 4),
        "avg_aether_mcc": round(avg_aether_mcc, 4),
        "avg_improvement_pct": round(avg_improvement, 2),
        "per_domain": domain_stats,
    }


def generate_markdown_report(results: dict[str, dict], missing: list[str]) -> str:
    """Generate the full markdown report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# AETHER Benchmark Comparison Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Overview",
        "",
        "This report compares AETHER's adaptive threshold governance against static thresholds "
        "across multiple process mining datasets. The key metric is **MCC (Matthews Correlation Coefficient)**, "
        "which balances true/false positives and negatives for the review gate decision.",
        "",
    ]

    # Missing datasets warning
    if missing:
        lines.extend([
            "### Missing Datasets",
            "",
            "The following datasets are not yet available:",
            "",
        ])
        for m in missing:
            lines.append(f"- {DATASETS[m]['label']} ({DATASETS[m]['domain']})")
        lines.append("")

    # Summary table
    lines.extend([
        "## Summary Results",
        "",
        generate_summary_table(results),
        "",
    ])

    # Aggregate statistics
    agg = generate_aggregate_stats(results)
    lines.extend([
        "## Aggregate Statistics",
        "",
        f"- **Total Datasets:** {agg['total_datasets']} / 8",
        f"- **Total Cases Evaluated:** {agg['total_cases']:,}",
        f"- **Average Model Accuracy:** {agg['avg_accuracy']:.1%}",
        f"- **Average Static MCC:** {agg['avg_static_mcc']:.4f}",
        f"- **Average AETHER MCC:** {agg['avg_aether_mcc']:.4f}",
        f"- **Average MCC Improvement:** {agg['avg_improvement_pct']:+.1f}%",
        "",
    ])

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
        generate_key_findings(results),
        "",
    ])

    # Per-domain analysis
    lines.extend([
        "## Domain Analysis",
        "",
        generate_domain_analysis(results),
        "",
    ])

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "### Evaluation Protocol",
        "",
        "1. **Ground Truth:** A case needs review if the model's next-activity prediction is wrong",
        "2. **Static Baseline:** Fixed threshold of 0.55 for the reviewGateAutoPass gate",
        "3. **AETHER Adaptive:** Dynamic threshold using v2 bidirectional formula:",
        "   - `effective_threshold = base * mode_factor * uncertainty_factor * calibration_factor`",
        "4. **Metrics:** MCC, F1, precision, recall, review burden (% of cases flagged)",
        "",
        "### Governance Modes",
        "",
        "| Mode | Factor | Description |",
        "|------|--------|-------------|",
        "| Flexible | 1.0 | Lower thresholds, fewer reviews |",
        "| Standard | 1.1 | Balanced (default) |",
        "| Strict | 1.2 | Higher thresholds, more reviews |",
        "",
        "Results shown use **standard** mode unless otherwise noted.",
        "",
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("AETHER BENCHMARK COMPARISON REPORT GENERATOR")
    print("=" * 60)

    # Load all available benchmarks
    results = {}
    missing = []

    print("\nLoading benchmark results...")
    for key, cfg in DATASETS.items():
        file_path = BENCHMARKS_DIR / cfg["file"]
        data = load_benchmark(file_path)

        if data:
            metrics = extract_metrics(data, mode="standard")
            results[key] = metrics
            print(f"  [OK] {cfg['label']}: {metrics['cases']} cases, MCC improvement: {metrics['improvement']:+.1f}%")
        else:
            missing.append(key)
            print(f"  [--] {cfg['label']}: not found")

    print(f"\nLoaded: {len(results)}/{len(DATASETS)} datasets")

    if not results:
        print("\nERROR: No benchmark results found. Cannot generate report.")
        return 1

    # Generate markdown report
    print("\nGenerating markdown report...")
    markdown = generate_markdown_report(results, missing)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "BENCHMARK_COMPARISON.md"
    with open(report_path, "w") as f:
        f.write(markdown)
    print(f"  Saved: {report_path}")

    # Generate JSON aggregate
    print("\nGenerating JSON summary...")
    aggregate = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "datasets_available": list(results.keys()),
        "datasets_missing": missing,
        "aggregate": generate_aggregate_stats(results),
        "per_dataset": {
            key: {
                "label": DATASETS[key]["label"],
                "domain": DATASETS[key]["domain"],
                "metrics": metrics,
            }
            for key, metrics in results.items()
        },
    }

    json_path = BENCHMARKS_DIR / "aggregate_results.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
