"""Experiment runner for the T-layer realizability study.

The runner produces the CSV, JSON, QASM, CNF, and ZIP artifacts used by the
paper

    Graph-Optimal T-Layering versus Replayable Local Realizability
    in Clifford+T Circuits

The default ``theory`` suite is intended to reproduce the paper-facing evidence:
circuit-derived benchmarks, graph-stress instances, and separation searches.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from t_layer_realizability_core import (
    ExperimentConfig,
    pack_artifacts,
    results_to_dataframe,
    graph_results_to_dataframe,
    run_graph_stress_instance,
    run_instance,
    self_test_once,
    write_json,
    write_summary,
    export_v11_theory_tables,
)


# ---------------------------------------------------------------------------
# Circuit suites
# ---------------------------------------------------------------------------
def default_suite(level: str = "standard") -> List[Dict[str, object]]:
    """Circuit-derived benchmarks.

    v11 adds two paper-oriented levels:
    - theory: moderate but broad, intended to feed theorem-evidence tables.
    - stress: larger n where circuit generation/transpilation is still plausible.
    """
    if level == "small":
        return [
            {"family": "qaoa_ring", "n_values": [4, 6, 8], "p_values": [1, 2], "depth_values": [1]},
            {"family": "qaoa_complete", "n_values": [4, 5], "p_values": [1], "depth_values": [1]},
            {"family": "random_ct", "n_values": [4, 6], "p_values": [1], "depth_values": [2, 4]},
            {"family": "brickwork_ct", "n_values": [6, 8], "p_values": [1], "depth_values": [2, 4]},
            {"family": "dense_phase", "n_values": [4, 5], "p_values": [1], "depth_values": [1, 2]},
            {"family": "clique_stress", "n_values": [4, 5], "p_values": [1], "depth_values": [1]},
            {"family": "grover", "n_values": [4, 5], "p_values": [1], "depth_values": [1]},
        ]
    if level == "theory":
        # Chosen to expose gap/separation behavior while keeping baselines feasible.
        return [
            {"family": "qaoa_ring", "n_values": [4, 8, 12, 16, 20, 24, 32], "p_values": [1, 2, 3], "depth_values": [1]},
            {"family": "qaoa_complete", "n_values": [4, 6, 8, 10, 12], "p_values": [1, 2], "depth_values": [1]},
            {"family": "random_ct", "n_values": [4, 8, 12, 16, 20], "p_values": [1], "depth_values": [2, 4, 6]},
            {"family": "random_interaction_ct", "n_values": [8, 12, 16, 20, 24], "p_values": [1], "depth_values": [2, 4]},
            {"family": "brickwork_ct", "n_values": [8, 16, 24, 32, 40], "p_values": [1], "depth_values": [2, 4]},
            {"family": "dense_phase", "n_values": [4, 6, 8, 10], "p_values": [1], "depth_values": [1, 2, 3]},
            {"family": "clique_stress", "n_values": [4, 6, 8, 10], "p_values": [1], "depth_values": [1, 2]},
            {"family": "qft_phase_like", "n_values": [4, 6, 8, 10], "p_values": [1], "depth_values": [1, 2]},
            {"family": "vqe_su2", "n_values": [4, 6, 8, 10], "p_values": [1, 2], "depth_values": [1]},
            {"family": "grover", "n_values": [4, 6, 8, 10, 12], "p_values": [1], "depth_values": [1]},
        ]
    if level == "extended":
        return [
            {"family": "qaoa_ring", "n_values": [4, 6, 8, 10, 12, 14, 16, 20], "p_values": [1, 2, 3], "depth_values": [1]},
            {"family": "qaoa_complete", "n_values": [4, 5, 6, 7, 8, 9, 10], "p_values": [1, 2], "depth_values": [1]},
            {"family": "random_ct", "n_values": [4, 6, 8, 10, 12, 14], "p_values": [1], "depth_values": [2, 4, 6, 8]},
            {"family": "random_interaction_ct", "n_values": [6, 8, 10, 12, 14], "p_values": [1], "depth_values": [2, 4, 6]},
            {"family": "brickwork_ct", "n_values": [8, 12, 16, 20, 24], "p_values": [1], "depth_values": [2, 4, 6]},
            {"family": "dense_phase", "n_values": [4, 5, 6, 7, 8, 9], "p_values": [1], "depth_values": [1, 2, 3]},
            {"family": "clique_stress", "n_values": [4, 5, 6, 7, 8, 9], "p_values": [1], "depth_values": [1, 2, 3]},
            {"family": "qft_phase_like", "n_values": [4, 5, 6, 7, 8], "p_values": [1], "depth_values": [1, 2]},
            {"family": "vqe_su2", "n_values": [4, 6, 8, 10], "p_values": [1, 2], "depth_values": [1]},
            {"family": "grover", "n_values": [4, 5, 6, 7, 8, 9, 10], "p_values": [1], "depth_values": [1]},
        ]
    if level == "large" or level == "stress":
        return [
            {"family": "qaoa_ring", "n_values": [8, 12, 16, 20, 24, 32], "p_values": [1, 2, 3, 4], "depth_values": [1]},
            {"family": "qaoa_complete", "n_values": [6, 8, 10, 12], "p_values": [1, 2], "depth_values": [1]},
            {"family": "random_ct", "n_values": [8, 12, 16, 20, 24], "p_values": [1], "depth_values": [4, 8]},
            {"family": "random_interaction_ct", "n_values": [8, 12, 16, 20, 24], "p_values": [1], "depth_values": [4, 8]},
            {"family": "brickwork_ct", "n_values": [16, 24, 32, 40], "p_values": [1], "depth_values": [4, 8]},
            {"family": "dense_phase", "n_values": [8, 10, 12], "p_values": [1], "depth_values": [2, 3, 4]},
            {"family": "clique_stress", "n_values": [8, 10, 12], "p_values": [1], "depth_values": [2, 3, 4]},
            {"family": "qft_phase_like", "n_values": [8, 10, 12], "p_values": [1], "depth_values": [1, 2]},
            {"family": "grover", "n_values": [6, 8, 10, 12], "p_values": [1], "depth_values": [1]},
        ]
    # standard
    return [
        {"family": "qaoa_ring", "n_values": [4, 6, 8, 10, 12, 14, 16, 20], "p_values": [1, 2, 3], "depth_values": [1]},
        {"family": "qaoa_complete", "n_values": [4, 5, 6, 7, 8, 10, 12], "p_values": [1, 2], "depth_values": [1]},
        {"family": "random_ct", "n_values": [4, 6, 8, 10, 12, 16, 20], "p_values": [1], "depth_values": [2, 4, 6]},
        {"family": "random_interaction_ct", "n_values": [6, 8, 10, 12, 16, 20], "p_values": [1], "depth_values": [2, 4]},
        {"family": "brickwork_ct", "n_values": [8, 12, 16, 20, 24, 32], "p_values": [1], "depth_values": [2, 4]},
        {"family": "dense_phase", "n_values": [4, 5, 6, 7, 8, 10], "p_values": [1], "depth_values": [1, 2, 3]},
        {"family": "clique_stress", "n_values": [4, 5, 6, 7, 8, 10], "p_values": [1], "depth_values": [1, 2]},
        {"family": "qft_phase_like", "n_values": [4, 5, 6, 7, 8, 10], "p_values": [1], "depth_values": [1, 2]},
        {"family": "vqe_su2", "n_values": [4, 6, 8, 10], "p_values": [1, 2], "depth_values": [1]},
        {"family": "grover", "n_values": [4, 5, 6, 7, 8, 10, 12], "p_values": [1], "depth_values": [1]},
    ]


# ---------------------------------------------------------------------------
# Graph-stress suites
# ---------------------------------------------------------------------------
def graph_stress_suite(level: str = "standard") -> List[Dict[str, object]]:
    if level == "small":
        return [
            {"graph_family": "path", "n_values": [8, 12]},
            {"graph_family": "cycle", "n_values": [8, 12]},
            {"graph_family": "complete", "n_values": [4, 5, 6]},
            {"graph_family": "erdos_renyi", "n_values": [8, 10], "densities": [0.25]},
        ]
    if level == "theory":
        return [
            {"graph_family": "path", "n_values": [32, 64, 128, 256, 512]},
            {"graph_family": "cycle", "n_values": [32, 64, 128, 256]},
            {"graph_family": "grid", "n_values": [25, 36, 49, 64, 81, 100]},
            {"graph_family": "complete", "n_values": [8, 10, 12, 16, 20, 24, 32]},
            {"graph_family": "complete_bipartite", "n_values": [20, 40, 60, 80, 100]},
            {"graph_family": "erdos_renyi", "n_values": [32, 48, 64], "densities": [0.10, 0.15, 0.25]},
            {"graph_family": "barabasi", "n_values": [40, 64, 96]},
            {"graph_family": "watts_strogatz", "n_values": [40, 64, 96]},
        ]
    if level == "stress" or level == "large":
        # Known families go very large using closed-form certificates; generic
        # random families are capped to avoid accidental all-night SAT runs.
        return [
            {"graph_family": "path", "n_values": [64, 128, 256, 512, 1024]},
            {"graph_family": "cycle", "n_values": [64, 128, 256, 512]},
            {"graph_family": "grid", "n_values": [49, 64, 81, 100, 144, 196]},
            {"graph_family": "complete", "n_values": [10, 12, 16, 20, 24, 32, 40, 48]},
            {"graph_family": "complete_bipartite", "n_values": [40, 80, 120, 160, 200]},
            {"graph_family": "erdos_renyi", "n_values": [32, 48, 64], "densities": [0.10, 0.15, 0.25]},
            {"graph_family": "barabasi", "n_values": [40, 64, 96]},
            {"graph_family": "watts_strogatz", "n_values": [40, 64, 96]},
        ]
    if level == "extended":
        return [
            {"graph_family": "path", "n_values": [16, 32, 64, 96, 128, 256]},
            {"graph_family": "cycle", "n_values": [16, 32, 64, 128]},
            {"graph_family": "grid", "n_values": [16, 25, 36, 49, 64, 81]},
            {"graph_family": "complete", "n_values": [6, 8, 10, 12, 16, 20, 24, 32]},
            {"graph_family": "complete_bipartite", "n_values": [12, 20, 30, 40, 60]},
            {"graph_family": "erdos_renyi", "n_values": [16, 24, 32, 40, 48], "densities": [0.15, 0.25, 0.35]},
            {"graph_family": "barabasi", "n_values": [24, 40, 64, 96]},
            {"graph_family": "watts_strogatz", "n_values": [24, 40, 64, 96]},
        ]
    # standard
    return [
        {"graph_family": "path", "n_values": [16, 32, 64, 128, 256]},
        {"graph_family": "cycle", "n_values": [16, 32, 64, 128]},
        {"graph_family": "grid", "n_values": [16, 25, 36, 49, 64, 100]},
        {"graph_family": "complete", "n_values": [6, 8, 10, 12, 16, 20, 24, 32]},
        {"graph_family": "complete_bipartite", "n_values": [12, 20, 40, 60, 100]},
        {"graph_family": "erdos_renyi", "n_values": [16, 24, 32, 48], "densities": [0.15, 0.25]},
        {"graph_family": "barabasi", "n_values": [24, 40, 64]},
        {"graph_family": "watts_strogatz", "n_values": [24, 40, 64]},
    ]


# ---------------------------------------------------------------------------
# Helpers for incremental output
# ---------------------------------------------------------------------------
def _write_graph_stress_outputs(out: Path, rows: list, errors: list) -> pd.DataFrame:
    df = graph_results_to_dataframe(rows)
    df.to_csv(out / "graph_stress_metrics.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / "graph_stress_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["name", "graph_family", "n", "density", "error"]).to_csv(out / "graph_stress_errors.csv", index=False)
    if not df.empty:
        df.groupby("graph_family").agg(
            instances=("name", "count"),
            max_vertices=("vertices", "max"),
            max_edges=("edges", "max"),
            max_treewidth=("treewidth_minfill", "max"),
            max_layers=("certified_graph_layers", "max"),
            mean_sat_seconds=("sat_seconds", "mean"),
            max_sat_seconds=("sat_seconds", "max"),
            mean_clauses=("sat_clauses", "mean"),
            max_clauses=("sat_clauses", "max"),
            solving_modes=("solving_mode", lambda x: ";".join(sorted(set(map(str, x))))),
        ).reset_index().to_csv(out / "graph_stress_summary.csv", index=False)
        numeric_cols = [c for c in ["vertices", "edges", "density", "treewidth_minfill", "certified_graph_layers", "sat_seconds", "sat_variables", "sat_clauses"] if c in df.columns]
        if numeric_cols:
            df[numeric_cols].corr(numeric_only=True).to_csv(out / "graph_stress_correlation_matrix.csv")
        df[[c for c in ["name", "graph_family", "vertices", "edges", "density", "treewidth_minfill", "certified_graph_layers", "sat_seconds", "sat_variables", "sat_clauses", "certificate_valid", "solving_mode", "stress_role", "scalability_interpretation"] if c in df.columns]].to_csv(out / "scalability_table.csv", index=False)
        df[[c for c in ["name", "graph_family", "vertices", "treewidth_minfill", "certified_graph_layers", "solving_mode", "stress_role", "scalability_interpretation"] if c in df.columns]].to_csv(out / "graph_stress_interpretation.csv", index=False)
    return df


def _write_separation_outputs(out: Path, rows: list, errors: list) -> pd.DataFrame:
    df = results_to_dataframe(rows) if rows else pd.DataFrame(columns=["name", "family", "n_qubits", "construction_gap", "graph_treewidth_minfill", "t_count"])
    df.to_csv(out / "separation_search_metrics.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / "separation_search_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["name", "family", "n", "seed", "depth", "p", "error"]).to_csv(out / "separation_search_errors.csv", index=False)
    if not df.empty:
        df["positive_gap"] = df["construction_gap"].fillna(0) > 0
        df.groupby(["family", "n_qubits"]).agg(
            trials=("name", "count"),
            separation_witnesses=("positive_gap", "sum"),
            separation_probability=("positive_gap", "mean"),
            mean_gap=("construction_gap", "mean"),
            max_gap=("construction_gap", "max"),
            mean_treewidth=("graph_treewidth_minfill", "mean"),
            max_treewidth=("graph_treewidth_minfill", "max"),
            mean_t_count=("t_count", "mean"),
            max_t_count=("t_count", "max"),
        ).reset_index().to_csv(out / "separation_probability_by_n.csv", index=False)
        df.groupby("family").agg(
            trials=("name", "count"),
            separation_witnesses=("positive_gap", "sum"),
            separation_probability=("positive_gap", "mean"),
            mean_gap=("construction_gap", "mean"),
            max_gap=("construction_gap", "max"),
            mean_treewidth=("graph_treewidth_minfill", "mean"),
            max_treewidth=("graph_treewidth_minfill", "max"),
        ).reset_index().to_csv(out / "gap_distribution_by_family.csv", index=False)
        cols = [c for c in ["name", "family", "n_qubits", "t_count", "graph_treewidth_minfill", "certified_graph_layers", "constructive_t_depth", "construction_gap", "retiming_improvement", "separation_witness", "sat_variables", "sat_clauses", "sat_seconds", "artifacts_dir"] if c in df.columns]
        df[cols].sort_values(["construction_gap", "t_count"], ascending=[False, False]).to_csv(out / "separation_witness_candidates_ranked.csv", index=False)
        # Treewidth vs gap table for scatter plots / correlation.
        tw_cols = [c for c in ["name", "family", "n_qubits", "t_count", "graph_edges", "graph_treewidth_minfill", "certified_graph_layers", "construction_gap", "sat_seconds", "sat_clauses"] if c in df.columns]
        df[tw_cols].to_csv(out / "treewidth_vs_gap.csv", index=False)
        corr_cols = [c for c in ["n_qubits", "t_count", "graph_edges", "graph_treewidth_minfill", "certified_graph_layers", "constructive_t_depth", "construction_gap", "sat_variables", "sat_clauses", "sat_seconds"] if c in df.columns]
        if corr_cols:
            df[corr_cols].corr(numeric_only=True).to_csv(out / "separation_correlation_matrix.csv")
        write_json(out / "separation_search_summary.json", {
            "trials": int(len(df)),
            "families": sorted(map(str, df["family"].unique())) if "family" in df else [],
            "separation_witnesses": int(df["positive_gap"].sum()),
            "separation_probability": float(df["positive_gap"].mean()),
            "max_gap": int(df["construction_gap"].max()) if "construction_gap" in df else None,
            "interpretation": "A positive construction gap is an empirical witness that the graph optimum is not reached by the conservative local-commutation constructor for that instance.",
        })
    return df


# ---------------------------------------------------------------------------
# Main experiment routines
# ---------------------------------------------------------------------------
def run_suite(
    out_dir: str,
    seed: int = 42,
    suite_level: str = "standard",
    use_zx: bool = False,
    no_maxsat: bool = False,
    clean: bool = False,
    pack: bool = True,
    continue_on_error: bool = True,
    graph_stress: bool = True,
    separation_search: bool = True,
    separation_seeds: int = 20,
    max_supplemental_equivalence_qubits: int = 12,
    max_external_baseline_qubits: int = 16,
    max_qiskit_baseline_qubits: int = 24,
) -> pd.DataFrame:
    out = Path(out_dir)
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        out_dir=str(out),
        seed=seed,
        use_zx_normalization=use_zx,
        use_maxsat_refinement=not no_maxsat,
        max_supplemental_equivalence_qubits=max_supplemental_equivalence_qubits,
        max_qcec_qubits=max_supplemental_equivalence_qubits,
        max_random_equivalence_qubits=max_supplemental_equivalence_qubits,
        max_external_baseline_qubits=max_external_baseline_qubits,
        max_qiskit_baseline_qubits=max_qiskit_baseline_qubits,
    )
    write_json(out / "experiment_config.json", cfg)
    write_json(out / "methodology.json", {
        "central_claim": "Certified graph-level optimum plus replayable local retiming equivalence; construction gap quantifies failure to realize graph optimum by the conservative constructor.",
        "suite_level": suite_level,
        "separation_search_enabled": separation_search,
        "graph_stress_enabled": graph_stress,
        "maxsat_note": "MaxSAT is not needed for graph-stress chromatic feasibility and is disabled there by default.",
        "guardrails": {
            "max_supplemental_equivalence_qubits": max_supplemental_equivalence_qubits,
            "max_external_baseline_qubits": max_external_baseline_qubits,
            "max_qiskit_baseline_qubits": max_qiskit_baseline_qubits,
            "large_n_claim": "For large n, the formal replay certificate is the equivalence guarantee; supplemental equivalence and optional baselines may be skipped and reported explicitly.",
        },
    })

    rows = []
    errors = []
    for block in default_suite(suite_level):
        fam = str(block["family"])
        for n in block["n_values"]:  # type: ignore[index]
            for p in block["p_values"]:  # type: ignore[index]
                for depth in block["depth_values"]:  # type: ignore[index]
                    name = f"{fam}_n{n}_p{p}_d{depth}"
                    print(f"[run] {name}")
                    try:
                        rows.append(run_instance(name, fam, int(n), cfg, depth=int(depth), p=int(p)))
                    except KeyboardInterrupt:
                        print(f"[interrupt] {name}")
                        df = results_to_dataframe(rows)
                        df.to_csv(out / "qip_suite_metrics.csv", index=False)
                        write_summary(df, out)
                        export_v11_theory_tables(out)
                        if pack:
                            pack_artifacts(out, "qip_tlayer_artifacts.zip")
                        return df
                    except Exception as exc:
                        print(f"[error] {name}: {exc}")
                        err = {"name": name, "family": fam, "n": int(n), "p": int(p), "depth": int(depth), "error": str(exc)}
                        errors.append(err)
                        write_json(out / name / "error.json", err)
                        if not continue_on_error:
                            raise
    df = results_to_dataframe(rows)
    df.to_csv(out / "qip_suite_metrics.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / "errors.csv", index=False)
    else:
        pd.DataFrame(columns=["name", "family", "n", "p", "depth", "error"]).to_csv(out / "errors.csv", index=False)
    summary = write_summary(df, out)
    summary["errors"] = len(errors)
    write_json(out / "summary.json", summary)

    if graph_stress:
        run_graph_stress(str(out), seed=seed, suite_level=suite_level, no_maxsat=True, continue_on_error=continue_on_error)
    if separation_search:
        run_separation_search(
            str(out), seed=seed, suite_level=suite_level, trials_per_family_n=separation_seeds,
            no_maxsat=no_maxsat, continue_on_error=continue_on_error,
            max_supplemental_equivalence_qubits=max_supplemental_equivalence_qubits,
            max_external_baseline_qubits=max_external_baseline_qubits,
            max_qiskit_baseline_qubits=max_qiskit_baseline_qubits,
        )
    export_v11_theory_tables(out)
    export_v11_master_theory_dashboard(out)
    if pack:
        pack_artifacts(out, "qip_tlayer_artifacts.zip")
    print(f"[done] {out / 'qip_suite_metrics.csv'}")
    return df


def run_graph_stress(out_dir: str, seed: int = 42, suite_level: str = "standard", no_maxsat: bool = True, continue_on_error: bool = True) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    write_json(out / "graph_stress_methodology.json", {
        "complete_graph_issue": "Generic SAT may spend a long time proving K_n is not (n-1)-colorable; this is symmetric pigeonhole UNSAT hardness, not a meaningful circuit-scaling signal.",
        "v11_policy": "Use closed-form certificates for graph families with known chromatic number and generic SAT for capped unstructured families.",
        "large_n_policy": "Known families scale to hundreds/thousands of vertices; random graph families are capped because exact coloring can become a standalone hard problem.",
        "maxsat_policy": "MaxSAT layer-sum refinement is disabled for graph stress by default.",
    })
    use_maxsat = not no_maxsat
    for block in graph_stress_suite(suite_level):
        gf = str(block["graph_family"])
        densities = block.get("densities", [0.25])  # type: ignore[assignment]
        for n in block["n_values"]:  # type: ignore[index]
            for density in densities:  # type: ignore[union-attr]
                name = f"{gf}_n{n}_rho{str(density).replace('.', 'p')}"
                print(f"[graph] {name}")
                try:
                    rows.append(run_graph_stress_instance(name, gf, int(n), str(out), seed=seed, density=float(density), use_maxsat=use_maxsat))
                    _write_graph_stress_outputs(out, rows, errors)
                except KeyboardInterrupt:
                    print(f"[graph-interrupted] {name}")
                    err = {"name": name, "graph_family": gf, "n": int(n), "density": float(density), "error": "KeyboardInterrupt"}
                    errors.append(err)
                    write_json(out / "graph_stress" / name / "error.json", err)
                    df = _write_graph_stress_outputs(out, rows, errors)
                    export_v11_theory_tables(out)
                    return df
                except Exception as exc:
                    print(f"[graph-error] {name}: {exc}")
                    err = {"name": name, "graph_family": gf, "n": int(n), "density": float(density), "error": str(exc)}
                    errors.append(err)
                    write_json(out / "graph_stress" / name / "error.json", err)
                    _write_graph_stress_outputs(out, rows, errors)
                    if not continue_on_error:
                        raise
    df = _write_graph_stress_outputs(out, rows, errors)
    export_v11_theory_tables(out)
    return df


def separation_search_suite(level: str = "standard") -> List[Dict[str, Any]]:
    """Theory-driven randomized/structured search for L_c > L* witnesses."""
    if level == "small":
        return [
            {"family": "random_ct", "n_values": [4, 6], "depth_values": [2, 4], "p_values": [1]},
            {"family": "random_interaction_ct", "n_values": [6, 8], "depth_values": [2], "p_values": [1]},
            {"family": "qaoa_complete", "n_values": [4, 6], "depth_values": [1], "p_values": [1]},
        ]
    if level == "theory":
        return [
            {"family": "random_ct", "n_values": list(range(4, 17, 2)), "depth_values": [2, 4, 6, 8], "p_values": [1]},
            {"family": "random_interaction_ct", "n_values": [6, 8, 10, 12, 14, 16, 20], "depth_values": [2, 4, 6], "p_values": [1]},
            {"family": "brickwork_ct", "n_values": [8, 12, 16, 20, 24, 32], "depth_values": [2, 4, 6], "p_values": [1]},
            {"family": "dense_phase", "n_values": [4, 6, 8, 10, 12], "depth_values": [1, 2, 3], "p_values": [1]},
            {"family": "clique_stress", "n_values": [4, 6, 8, 10, 12], "depth_values": [1, 2, 3], "p_values": [1]},
            {"family": "qaoa_complete", "n_values": [4, 6, 8, 10, 12], "depth_values": [1], "p_values": [1, 2]},
            {"family": "qft_phase_like", "n_values": [4, 6, 8, 10, 12], "depth_values": [1, 2], "p_values": [1]},
            {"family": "grover", "n_values": [4, 6, 8, 10, 12], "depth_values": [1], "p_values": [1]},
        ]
    # standard/stress are bounded to keep runtime controllable.
    return [
        {"family": "random_ct", "n_values": [4, 6, 8, 10, 12, 14], "depth_values": [2, 4, 6], "p_values": [1]},
        {"family": "random_interaction_ct", "n_values": [6, 8, 10, 12, 16], "depth_values": [2, 4], "p_values": [1]},
        {"family": "brickwork_ct", "n_values": [8, 12, 16, 20, 24], "depth_values": [2, 4], "p_values": [1]},
        {"family": "dense_phase", "n_values": [4, 6, 8, 10], "depth_values": [1, 2, 3], "p_values": [1]},
        {"family": "clique_stress", "n_values": [4, 6, 8, 10], "depth_values": [1, 2], "p_values": [1]},
        {"family": "qaoa_complete", "n_values": [4, 6, 8, 10, 12], "depth_values": [1], "p_values": [1, 2]},
        {"family": "qft_phase_like", "n_values": [4, 6, 8, 10], "depth_values": [1, 2], "p_values": [1]},
        {"family": "grover", "n_values": [4, 6, 8, 10, 12], "depth_values": [1], "p_values": [1]},
    ]


def run_separation_search(
    out_dir: str,
    seed: int = 42,
    suite_level: str = "standard",
    trials_per_family_n: int = 20,
    no_maxsat: bool = False,
    continue_on_error: bool = True,
    max_supplemental_equivalence_qubits: int = 12,
    max_external_baseline_qubits: int = 16,
    max_qiskit_baseline_qubits: int = 24,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdir = out / "separation_search"
    sdir.mkdir(parents=True, exist_ok=True)
    write_json(out / "separation_search_methodology.json", {
        "goal": "Search for empirical witnesses of graph-optimal vs locally constructible T-layer separation, L_c > L*.",
        "trials_per_family_n": trials_per_family_n,
        "suite_level": suite_level,
        "claim_scope": "Witnesses are relative to the implemented conservative local-commutation constructor, not impossibility under all possible circuit synthesis procedures.",
        "guardrails": {
            "max_supplemental_equivalence_qubits": max_supplemental_equivalence_qubits,
            "max_external_baseline_qubits": max_external_baseline_qubits,
            "max_qiskit_baseline_qubits": max_qiskit_baseline_qubits,
        },
    })
    rows = []
    errors = []
    blocks = separation_search_suite(suite_level)
    for block in blocks:
        fam = str(block["family"])
        for n in block["n_values"]:
            for depth in block["depth_values"]:
                for p in block["p_values"]:
                    for t in range(trials_per_family_n):
                        trial_seed = seed + 100000 * t + 1000 * int(n) + 37 * int(depth) + 13 * int(p)
                        name = f"sep_{fam}_n{n}_p{p}_d{depth}_s{trial_seed}"
                        print(f"[sep] {name}")
                        try:
                            cfg = ExperimentConfig(
                                out_dir=str(sdir),
                                seed=trial_seed,
                                use_zx_normalization=False,
                                use_maxsat_refinement=not no_maxsat,
                                # Keep equivalence checks bounded in randomized search.
                                verify_equivalence_up_to_qubits=min(6, max_supplemental_equivalence_qubits),
                                random_equivalence_tests=16,
                                max_supplemental_equivalence_qubits=max_supplemental_equivalence_qubits,
                                max_qcec_qubits=max_supplemental_equivalence_qubits,
                                max_random_equivalence_qubits=max_supplemental_equivalence_qubits,
                                max_external_baseline_qubits=max_external_baseline_qubits,
                                max_qiskit_baseline_qubits=max_qiskit_baseline_qubits,
                            )
                            rows.append(run_instance(name, fam, int(n), cfg, depth=int(depth), p=int(p)))
                            _write_separation_outputs(out, rows, errors)
                        except KeyboardInterrupt:
                            print(f"[sep-interrupted] {name}")
                            err = {"name": name, "family": fam, "n": int(n), "seed": int(trial_seed), "depth": int(depth), "p": int(p), "error": "KeyboardInterrupt"}
                            errors.append(err)
                            df = _write_separation_outputs(out, rows, errors)
                            export_v11_theory_tables(out)
                            return df
                        except Exception as exc:
                            print(f"[sep-error] {name}: {exc}")
                            err = {"name": name, "family": fam, "n": int(n), "seed": int(trial_seed), "depth": int(depth), "p": int(p), "error": str(exc)}
                            errors.append(err)
                            write_json(sdir / name / "error.json", err)
                            _write_separation_outputs(out, rows, errors)
                            if not continue_on_error:
                                raise
    df = _write_separation_outputs(out, rows, errors)
    export_v11_theory_tables(out)
    return df


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def export_v11_master_theory_dashboard(out_dir: str | Path) -> None:
    """Collect the main theorem-facing evidence files into one compact dashboard."""
    out = Path(out_dir)
    rows: List[Dict[str, Any]] = []
    metrics = _safe_read_csv(out / "qip_suite_metrics.csv")
    graph = _safe_read_csv(out / "graph_stress_metrics.csv")
    sep = _safe_read_csv(out / "separation_search_metrics.csv")
    if not metrics.empty:
        rows.append({
            "evidence_block": "circuit_suite",
            "instances": len(metrics),
            "max_n": metrics.get("n_qubits", pd.Series(dtype=float)).max(),
            "max_t_count": metrics.get("t_count", pd.Series(dtype=float)).max(),
            "max_treewidth": metrics.get("graph_treewidth_minfill", pd.Series(dtype=float)).max(),
            "max_gap": metrics.get("construction_gap", pd.Series(dtype=float)).max(),
            "positive_gap_cases": int((metrics.get("construction_gap", pd.Series(dtype=float)).fillna(0) > 0).sum()) if "construction_gap" in metrics else None,
            "role": "main circuit-derived evidence",
        })
    if not graph.empty:
        rows.append({
            "evidence_block": "graph_stress",
            "instances": len(graph),
            "max_n": graph.get("vertices", pd.Series(dtype=float)).max(),
            "max_t_count": None,
            "max_treewidth": graph.get("treewidth_minfill", pd.Series(dtype=float)).max(),
            "max_gap": None,
            "positive_gap_cases": None,
            "role": "controlled structural scalability evidence",
        })
    if not sep.empty:
        rows.append({
            "evidence_block": "separation_search",
            "instances": len(sep),
            "max_n": sep.get("n_qubits", pd.Series(dtype=float)).max(),
            "max_t_count": sep.get("t_count", pd.Series(dtype=float)).max(),
            "max_treewidth": sep.get("graph_treewidth_minfill", pd.Series(dtype=float)).max(),
            "max_gap": sep.get("construction_gap", pd.Series(dtype=float)).max(),
            "positive_gap_cases": int((sep.get("construction_gap", pd.Series(dtype=float)).fillna(0) > 0).sum()) if "construction_gap" in sep else None,
            "role": "targeted witness search for separation theorem",
        })
    if rows:
        pd.DataFrame(rows).to_csv(out / "v11_theory_dashboard.csv", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts_qip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suite", choices=["small", "standard", "extended", "large", "theory", "stress"], default="standard")
    ap.add_argument("--use-zx", action="store_true", help="Use optional ZX normalization before encoding. Default is off for cleanest proof story.")
    ap.add_argument("--no-maxsat", action="store_true", help="Disable MaxSAT layer-sum refinement in circuit/separation runs.")
    ap.add_argument("--graph-maxsat", action="store_true", help="Enable MaxSAT refinement in graph-stress experiments. Not recommended.")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--graph-stress", action="store_true", default=True, help="Run graph-only stress benchmarks. Enabled by default.")
    ap.add_argument("--no-graph-stress", action="store_true")
    ap.add_argument("--separation-search", action="store_true", default=True, help="Run targeted L_c > L* witness search. Enabled by default.")
    ap.add_argument("--no-separation-search", action="store_true")
    ap.add_argument("--separation-seeds", type=int, default=20, help="Trials per family/n/depth/p setting in separation search.")
    ap.add_argument("--only-graph-stress", action="store_true")
    ap.add_argument("--only-separation-search", action="store_true")
    ap.add_argument("--max-supplemental-equivalence-qubits", type=int, default=12, help="Above this n, skip QCEC/statevector supplemental checks and rely on formal replay proof.")
    ap.add_argument("--max-external-baseline-qubits", type=int, default=16, help="Above this n, skip optional external baselines tket/PyZX.")
    ap.add_argument("--max-qiskit-baseline-qubits", type=int, default=24, help="Above this n, skip multi-level Qiskit baseline.")
    args = ap.parse_args()

    if args.self_test:
        df = self_test_once(str(Path(args.outdir) / "selftest"))
        print(df)
    if args.only_graph_stress:
        run_graph_stress(args.outdir, args.seed, args.suite, no_maxsat=(not args.graph_maxsat), continue_on_error=True)
        export_v11_master_theory_dashboard(args.outdir)
        return
    if args.only_separation_search:
        run_separation_search(
            args.outdir, args.seed, args.suite, args.separation_seeds, no_maxsat=args.no_maxsat, continue_on_error=True,
            max_supplemental_equivalence_qubits=args.max_supplemental_equivalence_qubits,
            max_external_baseline_qubits=args.max_external_baseline_qubits,
            max_qiskit_baseline_qubits=args.max_qiskit_baseline_qubits,
        )
        export_v11_master_theory_dashboard(args.outdir)
        return

    run_suite(
        args.outdir,
        args.seed,
        args.suite,
        args.use_zx,
        args.no_maxsat,
        args.clean,
        pack=not args.no_pack,
        graph_stress=(args.graph_stress and not args.no_graph_stress),
        separation_search=(args.separation_search and not args.no_separation_search),
        separation_seeds=args.separation_seeds,
        max_supplemental_equivalence_qubits=args.max_supplemental_equivalence_qubits,
        max_external_baseline_qubits=args.max_external_baseline_qubits,
        max_qiskit_baseline_qubits=args.max_qiskit_baseline_qubits,
    )


if __name__ == "__main__":
    main()
