# Graph-Optimal T-Layering versus Replayable Local Realizability

**Author:** Csaba Biró  
**Paper:** *Graph-Optimal T-Layering versus Replayable Local Realizability in Clifford+T Circuits*

This repository contains the experiment code and artifact-generation pipeline for the paper.  
The code is organized around the paper's main distinction:

- `L*(C)`: the SAT-certified graph-optimal T-layering value of the dependency graph;
- `L_A(C)`: the T-depth produced by the fixed replayable local-commutation constructor;
- `Delta(C) = L_A(C) - L*(C)`: the observed construction gap.

The implementation is not intended to be a competitive quantum circuit optimizer. Its purpose is to make the graph-level optimum, the replayable local construction, and the observed realizability gap explicit and reproducible.

## Files

- `t_layer_realizability_core.py`  
  Core implementation: circuit families, dependency-graph construction, SAT coloring certificates, replayable local retiming, obstruction logging, optional baselines, and artifact export.

- `run_experiments.py`  
  Command-line runner for the circuit suite, separation search, and graph-stress experiments.

- `README.md`  
  This file.

## Requirements

The core experiments require Python 3.10+ and the following packages:

```bash
pip install qiskit networkx numpy pandas python-sat
```

Optional packages used for supplemental baselines/checks:

```bash
pip install pyzx pytket pytket-qiskit mqt.qcec
```

The paper's main replay-based equivalence claim does not depend on these optional tools. They are reported as supplemental evidence or baselines when available.

## Recommended paper run

```bash
python run_experiments.py \
  --outdir artifacts_qip \
  --suite theory \
  --clean \
  --separation-seeds 20
```

For large instances, supplemental statevector/QCEC checks and external baselines can be skipped by design while the replay certificate is still produced:

```bash
python run_experiments.py \
  --outdir artifacts_qip \
  --suite theory \
  --clean \
  --separation-seeds 20 \
  --max-supplemental-equivalence-qubits 12 \
  --max-external-baseline-qubits 16 \
  --max-qiskit-baseline-qubits 24
```

## Quick tests

Run only graph-stress examples:

```bash
python run_experiments.py \
  --outdir artifacts_graph_test \
  --suite small \
  --only-graph-stress
```

Run only a small separation search:

```bash
python run_experiments.py \
  --outdir artifacts_sep_test \
  --suite small \
  --only-separation-search \
  --separation-seeds 2
```

Run the built-in self-test:

```bash
python run_experiments.py --self-test --outdir artifacts_selftest
```

## Main output files

The full run produces a directory such as `artifacts_qip/` containing, among others:

- `qip_suite_metrics.csv`
- `construction_gap_summary.csv`
- `equivalence_summary.csv`
- `baseline_summary.csv`
- `separation_examples.csv`
- `separation_search_metrics.csv`
- `separation_probability_by_n.csv`
- `treewidth_vs_gap.csv`
- `graph_stress_metrics.csv`
- `graph_stress_summary.csv`
- `v11_theory_dashboard.csv`
- per-instance JSON certificates and QASM files
- `qip_tlayer_artifacts.zip`

## Interpretation of the artifacts

The SAT layer certifies a graph-level T-layer optimum for the extracted dependency graph. The circuit-level equivalence claim for the constructed retiming is supplied by a replayable sequence of locally sound adjacent commutations.

Optional QCEC/statevector checks and external baselines are useful diagnostics, but they are not part of the formal replay certificate. For large instances, these supplemental checks may be explicitly skipped using the guardrails above.

## Reproducibility notes

- Randomized circuit families use deterministic seeds.
- Each accepted local retiming step is recorded in `retiming_certificate.json`.
- Failed local swaps are recorded and summarized as obstruction evidence.
- Graph-only stress instances are not circuit-level construction-gap witnesses; they isolate the graph-coloring certification layer.
