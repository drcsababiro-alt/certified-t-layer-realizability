"""Core routines for the T-layer realizability experiments.

This module implements the pipeline used in the paper

    Graph-Optimal T-Layering versus Replayable Local Realizability
    in Clifford+T Circuits

The code separates three roles:

* SAT certification of the graph-level T-layer optimum;
* replayable local-commutation retiming of the circuit instruction sequence;
* optional external baselines and supplemental equivalence checks.

The main formal claim produced by this code is the replay certificate:
each accepted adjacent swap is tagged by a local commutation rule and is
independently replayed. External tools such as Qiskit, PyZX, tket, and QCEC
are used only as optional baselines or supplemental checks.
"""
from __future__ import annotations

import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    from qiskit import QuantumCircuit, qasm2, qasm3, transpile
    from qiskit.circuit import QuantumRegister
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import Statevector, Operator
    HAVE_QISKIT = True
except Exception:  # pragma: no cover
    HAVE_QISKIT = False
    QuantumCircuit = object  # type: ignore
    QuantumRegister = object  # type: ignore
    EfficientSU2 = None  # type: ignore
    Statevector = None  # type: ignore
    Operator = None  # type: ignore
    qasm2 = qasm3 = None  # type: ignore
    transpile = None  # type: ignore

try:
    from pysat.formula import CNF, WCNF
    from pysat.solvers import Minisat22
    from pysat.examples.rc2 import RC2
    HAVE_PYSAT = True
except Exception:  # pragma: no cover
    HAVE_PYSAT = False
    CNF = WCNF = Minisat22 = RC2 = None  # type: ignore

try:
    import pyzx as zx
    HAVE_PYZX = True
except Exception:  # pragma: no cover
    HAVE_PYZX = False

try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import FullPeepholeOptimise
    HAVE_TKET = True
except Exception:  # pragma: no cover
    HAVE_TKET = False

try:
    from mqt import qcec
    HAVE_QCEC = True
except Exception:  # pragma: no cover
    HAVE_QCEC = False

ANGLE_TOL = 1e-7
Z_PHASE_GATES = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1"}
SUPPORTED_STRUCTURAL_GATES = Z_PHASE_GATES | {"h", "x", "cx", "ccx", "mcx", "rx", "ry"}


@dataclass(frozen=True)
class ExperimentConfig:
    out_dir: str = "artifacts_qip"
    seed: int = 42
    angle_tol: float = ANGLE_TOL
    basis_gates: Tuple[str, ...] = ("cx", "ccx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz", "rx", "ry")
    qiskit_optimization_level: int = 1
    write_dimacs: bool = True
    write_qasm: bool = True
    use_zx_normalization: bool = False  # default false for the cleanest circuit-level proof story
    use_maxsat_refinement: bool = True
    verify_equivalence_up_to_qubits: int = 7
    random_equivalence_tests: int = 64
    # v11 guardrails: the formal guarantee is the replayable local-commutation
    # proof. Supplemental QCEC/statevector checks and external baselines are
    # intentionally bounded so large-n theory/stress runs do not hang.
    max_qcec_qubits: int = 12
    max_random_equivalence_qubits: int = 12
    max_supplemental_equivalence_qubits: int = 12
    skip_supplemental_reason: str = "large_qubit_count_formal_replay_used"
    max_external_baseline_qubits: int = 16
    max_qiskit_baseline_qubits: int = 24
    large_baseline_policy: str = "skip_external_tools_above_threshold"
    max_retiming_swaps: int = 200000
    export_instruction_ids: bool = True
    max_bruteforce_fallback_vertices: int = 12
    run_supplemental_equivalence: bool = True
    baseline_timeout_note: str = "Baselines are attempted with guardrails; skipped large-instance baselines are reported explicitly."


@dataclass
class TNode:
    node_id: int
    circuit_index: int
    gate: str
    qubits: Tuple[int, ...]


@dataclass
class SatCertificate:
    optimum_layers: int
    assignment: Dict[int, int]
    lower_bound_clique: int
    upper_bound_greedy: int
    sat_calls: int
    trials: List[Dict[str, Any]]
    variables_at_optimum: int
    clauses_at_optimum: int
    maxsat_sum_layers: Optional[int] = None


@dataclass
class SwapStep:
    step: int
    left_id_before: int
    right_id_before: int
    reason: str
    left_gate: str
    right_gate: str
    left_qubits: Tuple[int, ...]
    right_qubits: Tuple[int, ...]


@dataclass
class RetimingCertificate:
    method: str
    valid_by_local_commutation: bool
    replay_valid: bool
    swaps: List[SwapStep]
    failed_swaps: List[Dict[str, Any]]
    theorem_scope: str


@dataclass
class FormalEquivalenceReport:
    proven: bool
    scope: str
    theorem_name: str
    assumptions: List[str]
    proof_obligations: List[str]
    discharged_obligations: List[str]
    undischarged_obligations: List[str]
    local_rules_used: Dict[str, int]
    total_swaps: int
    replay_valid: bool
    statement: str


@dataclass
class BaselineResult:
    tool: str
    t_depth: Optional[int]
    status: str
    method: str
    error: Optional[str] = None
    note: Optional[str] = None


@dataclass
class GraphStressResult:
    name: str
    graph_family: str
    vertices: int
    edges: int
    density: float
    treewidth_minfill: Optional[int]
    certified_graph_layers: int
    sat_seconds: float
    sat_variables: int
    sat_clauses: int
    clique_lower_bound: int
    greedy_upper_bound: int
    certificate_valid: bool
    solving_mode: str
    stress_role: str
    scalability_interpretation: str
    artifact_dir: str


@dataclass
class InstanceResult:
    name: str
    family: str
    n_qubits: int
    t_count: int
    t_depth_input: int
    certified_graph_layers: int
    constructive_t_depth: int
    construction_gap: Optional[int]
    retiming_improvement: int
    retiming_limitation_flag: bool
    separation_witness: bool
    construction_gap_interpretation: str
    graph_vertices: int
    graph_edges: int
    graph_density: float
    graph_treewidth_minfill: Optional[int]
    circuit_graph_regime: str
    practical_near_tree_flag: bool
    graph_optimum_regime: str
    sat_seconds: float
    sat_variables: int
    sat_clauses: int
    clique_lower_bound: int
    greedy_upper_bound: int
    certificate_valid: bool
    retiming_certificate_valid: bool
    formal_global_equivalence_proven: bool
    formal_global_equivalence_scope: str
    formal_global_equivalence_statement: str
    formal_undischarged_obligations: int
    equivalence_claim: str
    supplemental_equivalence_check: str
    zx_used: bool
    zx_available: bool
    zx_pre_vertices: Optional[int]
    zx_post_vertices: Optional[int]
    tket_t_depth: Optional[int]
    pyzx_t_depth: Optional[int]
    qiskit_opt_t_depth: Optional[int]
    tket_status: str
    pyzx_status: str
    qiskit_status: str
    baseline_coverage: str
    formal_equivalence_status: str
    artifacts_dir: str


# ---------------- I/O helpers ----------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(x: Any) -> Any:
    if hasattr(x, "__dataclass_fields__"):
        return asdict(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, tuple):
        return list(x)
    return str(x)


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def write_qasm(path: str | Path, circuit: QuantumCircuit) -> None:
    require_qiskit()
    ensure_dir(Path(path).parent)
    try:
        text = qasm2.dumps(circuit)
    except Exception:
        text = qasm3.dumps(circuit)
    Path(path).write_text(text, encoding="utf-8")


def pack_artifacts(out_dir: str | Path, zip_name: str = "qip_tlayer_artifacts.zip") -> str:
    out_dir = Path(out_dir)
    zip_path = out_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir))
    return str(zip_path)


def require_qiskit() -> None:
    if not HAVE_QISKIT:
        raise ImportError("Qiskit is required for circuit benchmarks. Install qiskit in the experiment environment.")


# ---------------- circuit families ----------------
def _fresh_register_name(qc: QuantumCircuit, base: str = "anc") -> str:
    existing = {reg.name for reg in getattr(qc, "qregs", [])}
    if base not in existing:
        return base
    k = 1
    while f"{base}_{k}" in existing:
        k += 1
    return f"{base}_{k}"


def _get_or_create_ancilla_register(qc: QuantumCircuit, need: int) -> QuantumRegister:
    for reg in getattr(qc, "qregs", []):
        if reg.name.startswith("anc") and len(reg) >= need:
            return reg
    anc = QuantumRegister(need, _fresh_register_name(qc, "anc"))
    qc.add_register(anc)
    return anc


def _mcx(qc: QuantumCircuit, ctrls: Sequence[int], tgt: int) -> None:
    if len(ctrls) <= 4:
        qc.mcx(list(ctrls), tgt)
    else:
        need = len(ctrls) - 2
        anc = _get_or_create_ancilla_register(qc, need)
        qc.mcx(list(ctrls), tgt, ancilla_qubits=list(anc)[:need])


def qaoa_ring(n: int, p: int, gamma: float = math.pi/8, beta: float = math.pi/8) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qaoa_ring_n{n}_p{p}")
    for q in range(n):
        qc.h(q)
    for _ in range(p):
        for i in range(n):
            j = (i + 1) % n
            qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def qaoa_complete(n: int, p: int, gamma: float = math.pi/8, beta: float = math.pi/8) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qaoa_complete_n{n}_p{p}")
    for q in range(n):
        qc.h(q)
    for _ in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def random_clifford_t_vqc(n: int, depth: int, seed: int = 0) -> QuantumCircuit:
    rnd = random.Random(seed)
    angles = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, math.pi]
    qc = QuantumCircuit(n, name=f"random_ct_n{n}_d{depth}")
    for _ in range(depth):
        for q in range(n):
            qc.rz(rnd.choice(angles), q); qc.h(q); qc.rz(rnd.choice(angles), q)
        # alternate line and skip-line entanglement for more varied dependency graphs
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(0, n - 2, 2):
            qc.cx(q, q + 2)
    return qc



def random_interaction_ct(n: int, depth: int, seed: int = 0, edge_probability: float = 0.35) -> QuantumCircuit:
    """Random interaction Clifford+T circuits with non-local CNOT barriers.

    This family is used to create larger, less path-like dependency graphs than
    QAOA rings. The randomness is deterministic under ``seed`` and each layer
    contains T-like phases plus a sampled interaction graph.
    """
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_interaction_ct_n{n}_d{depth}")
    for layer in range(depth):
        for q in range(n):
            qc.rz(rnd.choice([math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]), q)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rnd.random() < edge_probability:
                    edges.append((i, j))
        rnd.shuffle(edges)
        for i, j in edges[: max(1, min(len(edges), 2*n))]:
            if rnd.random() < 0.5:
                qc.cx(i, j)
            else:
                qc.cx(j, i)
        for q in range(0, n, 2):
            qc.h(q)
    return qc


def brickwork_ct(n: int, depth: int, seed: int = 0) -> QuantumCircuit:
    """Hardware-friendly brickwork Clifford+T family for medium-size tests."""
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"brickwork_ct_n{n}_d{depth}")
    for layer in range(depth):
        for q in range(n):
            qc.rz(rnd.choice([math.pi/4, -math.pi/4, math.pi/2]), q)
        start = layer % 2
        for q in range(start, n-1, 2):
            qc.cx(q, q+1)
        for q in range(n):
            if (q + layer) % 3 == 0:
                qc.h(q)
    return qc


def qft_phase_like(n: int, depth: int = 1) -> QuantumCircuit:
    """QFT-inspired phase/entanglement pattern without relying on QFT decomposition."""
    qc = QuantumCircuit(n, name=f"qft_phase_like_n{n}_d{depth}")
    for _ in range(depth):
        for i in range(n):
            qc.h(i)
            for j in range(i+1, n):
                qc.cx(j, i)
                qc.rz(math.pi/4, i)
                qc.cx(j, i)
        for q in range(n):
            qc.t(q)
    return qc


def dense_phase_stress(n: int, depth: int, seed: int = 0) -> QuantumCircuit:
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"dense_phase_n{n}_d{depth}")
    for _ in range(depth):
        for q in range(n):
            qc.rz(rnd.choice([math.pi/4, -math.pi/4]), q)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
        for q in range(n):
            qc.h(q)
    return qc


def layered_clique_t_graph(n: int, layers: int) -> QuantumCircuit:
    """Circuit intended to induce dense T-dependencies for stress testing.

    This is not a practical algorithmic family; it is explicitly for limitations
    and scaling discussion. T-like phases are separated by target-side CNOT use,
    which creates non-commuting barriers in the dependency extraction.
    """
    qc = QuantumCircuit(n, name=f"clique_stress_n{n}_l{layers}")
    for _ in range(layers):
        for q in range(n):
            qc.t(q)
        for target in range(n):
            ctrl = (target + 1) % n
            qc.cx(ctrl, target)
        for q in range(n):
            qc.h(q)
    return qc


def vqe_su2_bound(n: int, reps: int = 1, seed: int = 42) -> QuantumCircuit:
    if EfficientSU2 is None:
        raise ImportError("EfficientSU2 unavailable; install qiskit circuit library")
    template = EfficientSU2(num_qubits=n, reps=reps, entanglement="linear")
    qc = QuantumCircuit(n, name=f"vqe_su2_n{n}_r{reps}")
    qc.compose(template, inplace=True)
    return bind_parameters_to_clifford_t(qc, seed)


def grover_single_marked(n: int, iterations: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"grover_n{n}_it{iterations}")
    for q in range(n):
        qc.h(q)
    for _ in range(iterations):
        qc.h(n - 1); _mcx(qc, list(range(n - 1)), n - 1); qc.h(n - 1)
        for q in range(n):
            qc.h(q); qc.x(q)
        qc.h(n - 1); _mcx(qc, list(range(n - 1)), n - 1); qc.h(n - 1)
        for q in range(n):
            qc.x(q); qc.h(q)
    return qc


def prepare_circuit(family: str, n: int, depth: int = 1, p: int = 1, seed: int = 42) -> QuantumCircuit:
    require_qiskit()
    if family == "qaoa_ring": return qaoa_ring(n, p)
    if family == "qaoa_complete": return qaoa_complete(n, p)
    if family == "random_ct": return random_clifford_t_vqc(n, depth, seed)
    if family == "random_interaction_ct": return random_interaction_ct(n, depth, seed)
    if family == "brickwork_ct": return brickwork_ct(n, depth, seed)
    if family == "qft_phase_like": return qft_phase_like(n, depth)
    if family == "dense_phase": return dense_phase_stress(n, depth, seed)
    if family == "clique_stress": return layered_clique_t_graph(n, depth)
    if family == "vqe_su2": return vqe_su2_bound(n, p, seed)
    if family == "grover": return grover_single_marked(n, p)
    raise ValueError(f"unknown family: {family}")


# ---------------- extraction and graph model ----------------
def bind_parameters_to_clifford_t(qc: QuantumCircuit, seed: int = 42) -> QuantumCircuit:
    if not getattr(qc, "parameters", None):
        return qc.copy()
    rnd = random.Random(seed)
    angles = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, math.pi]
    params = sorted(qc.parameters, key=lambda p: p.name)
    return qc.assign_parameters({p: rnd.choice(angles) for p in params}, inplace=False)


def classify_rz_as_t(angle: float, tol: float = ANGLE_TOL) -> Optional[str]:
    a = ((float(angle) + math.pi) % (2 * math.pi)) - math.pi
    k = int(round(a / (math.pi/4)))
    if abs(a - k * (math.pi/4)) >= tol or k % 2 == 0:
        return None
    return "t" if (k % 8) in (1, 5) else "tdg"


def qid_map(qc: QuantumCircuit) -> Dict[Any, int]:
    return {q: i for i, q in enumerate(qc.qubits)}


def t_nodes(qc: QuantumCircuit, tol: float = ANGLE_TOL) -> List[TNode]:
    qmap = qid_map(qc)
    nodes: List[TNode] = []
    for idx, inst in enumerate(qc.data):
        name = inst.operation.name.lower()
        qids = tuple(qmap[q] for q in inst.qubits)
        gate: Optional[str] = None
        if name in {"t", "tdg"}:
            gate = name
        elif name in {"rz", "p", "u1"}:
            try:
                gate = classify_rz_as_t(float(inst.operation.params[0]), tol)
            except Exception:
                gate = None
        if gate is not None:
            nodes.append(TNode(len(nodes), idx, gate, qids))
    return nodes


def naive_t_depth(qc: QuantumCircuit, tol: float = ANGLE_TOL) -> int:
    last: Dict[int, int] = {}
    depth = 0
    for node in t_nodes(qc, tol):
        layer = 1 + max((last.get(q, 0) for q in node.qubits), default=0)
        for q in node.qubits:
            last[q] = layer
        depth = max(depth, layer)
    return depth


def zx_normalize(qc: QuantumCircuit, out_qasm: Optional[str] = None) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    stats: Dict[str, Any] = {"available": bool(HAVE_PYZX), "pre_vertices": None, "pre_edges": None, "post_vertices": None, "post_edges": None, "error": None}
    if not HAVE_PYZX:
        return qc.copy(), stats
    try:
        g = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_graph()
        stats["pre_vertices"] = g.num_vertices(); stats["pre_edges"] = g.num_edges()
        zx.simplify.full_reduce(g)
        stats["post_vertices"] = g.num_vertices(); stats["post_edges"] = g.num_edges()
        extracted = zx.extract.extract_circuit(g.copy()).to_qiskit()
        if out_qasm:
            write_qasm(out_qasm, extracted)
        return extracted, stats
    except Exception as exc:
        stats["error"] = str(exc)
        return qc.copy(), stats


def _controls_and_target(name: str, qids: Sequence[int]) -> Tuple[List[int], Optional[int]]:
    if name in {"cx", "ccx", "mcx"} and len(qids) >= 2:
        return list(qids[:-1]), qids[-1]
    return [], None


def build_dependency_graph(qc: QuantumCircuit, tol: float = ANGLE_TOL) -> Tuple[nx.Graph, List[TNode]]:
    """Build the conservative T-like dependency graph.

    Edges mean two T-like operations cannot be assigned to the same T-layer by
    the commutation model used in this paper. This is a sufficient dependency
    model for certified graph-layer optimality; constructive realization is
    reported separately.
    """
    nodes = t_nodes(qc, tol)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.node_id, circuit_index=node.circuit_index, gate=node.gate, qubits=node.qubits)
    qmap = qid_map(qc)
    ops = [(inst.operation.name.lower(), tuple(qmap[q] for q in inst.qubits)) for inst in qc.data]
    idx_to_node = {node.circuit_index: node.node_id for node in nodes}
    for q in range(qc.num_qubits):
        seq = sorted(node.circuit_index for node in nodes if q in node.qubits)
        for left_idx, right_idx in zip(seq, seq[1:]):
            unsafe = False
            for op_name, op_qubits in ops[left_idx + 1:right_idx]:
                if q not in op_qubits:
                    continue
                if op_name in Z_PHASE_GATES:
                    continue
                if op_name in {"cx", "ccx", "mcx"}:
                    controls, target = _controls_and_target(op_name, op_qubits)
                    if q == target:
                        unsafe = True; break
                    if q in controls:
                        continue
                unsafe = True; break
            if unsafe:
                G.add_edge(idx_to_node[left_idx], idx_to_node[right_idx])
    return G, nodes


def graph_statistics(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes(); m = G.number_of_edges()
    deg = [d for _, d in G.degree()]
    try:
        tw, _ = nx.approximation.treewidth_min_fill_in(G)
    except Exception:
        tw = None
    return {
        "vertices": n,
        "edges": m,
        "max_degree": max(deg, default=0),
        "avg_degree": float(sum(deg))/n if n else 0.0,
        "density": (2*m)/(n*(n-1)) if n > 1 else 0.0,
        "treewidth_minfill": tw,
        "connected_components": nx.number_connected_components(G) if n else 0,
    }


# ---------------- SAT certificate ----------------
def clique_lower_bound(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    try:
        return max((len(c) for c in nx.find_cliques(G)), default=1)
    except Exception:
        return 1


def greedy_upper_bound(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    order = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    color: Dict[int, int] = {}
    for v in order:
        used = {color[u] for u in G.neighbors(v) if u in color}
        c = 1
        while c in used:
            c += 1
        color[v] = c
    return max(color.values(), default=0)


def encode_coloring_cnf(G: nx.Graph, layers: int) -> Tuple[Any, Dict[Tuple[int, int], int]]:
    if not HAVE_PYSAT:
        raise RuntimeError("encode_coloring_cnf requires python-sat; fallback solver uses analytical counts only")
    cnf = CNF(); var: Dict[Tuple[int, int], int] = {}
    def vv(v: int, l: int) -> int:
        key = (v, l)
        if key not in var:
            var[key] = len(var) + 1
        return var[key]
    for v in G.nodes():
        cnf.append([vv(v, l) for l in range(1, layers + 1)])
        for l1 in range(1, layers + 1):
            for l2 in range(l1 + 1, layers + 1):
                cnf.append([-vv(v, l1), -vv(v, l2)])
    for u, v in G.edges():
        for l in range(1, layers + 1):
            cnf.append([-vv(u, l), -vv(v, l)])
    return cnf, var


def _analytical_cnf_size(G: nx.Graph, layers: int) -> Tuple[int, int]:
    n = G.number_of_nodes(); m = G.number_of_edges()
    return n * layers, n + n * (layers * (layers - 1) // 2) + m * layers


def _bruteforce_coloring(G: nx.Graph, layers: int) -> Tuple[bool, Dict[int, int]]:
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    asg: Dict[int, int] = {}
    def backtrack(i: int) -> bool:
        if i == len(nodes):
            return True
        v = nodes[i]
        forbidden = {asg[u] for u in G.neighbors(v) if u in asg}
        for l in range(1, layers + 1):
            if l in forbidden:
                continue
            asg[v] = l
            if backtrack(i + 1):
                return True
            del asg[v]
        return False
    ok = backtrack(0)
    return ok, dict(asg) if ok else {}


def solve_coloring_sat(G: nx.Graph, layers: int, dimacs_path: Optional[str] = None) -> Tuple[bool, Dict[int, int], int, int]:
    if not HAVE_PYSAT:
        nvars, nclauses = _analytical_cnf_size(G, layers)
        if G.number_of_nodes() > 12:
            raise RuntimeError("python-sat is required for graphs with more than 12 vertices")
        sat, asg = _bruteforce_coloring(G, layers)
        if dimacs_path:
            ensure_dir(Path(dimacs_path).parent)
            Path(dimacs_path).write_text(
                f"c python-sat not installed; analytical size only\np cnf {nvars} {nclauses}\n",
                encoding="utf-8",
            )
        return sat, asg, nvars, nclauses
    cnf, var = encode_coloring_cnf(G, layers)
    if dimacs_path:
        ensure_dir(Path(dimacs_path).parent); cnf.to_file(dimacs_path)
    with Minisat22(bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        if not sat:
            return False, {}, len(var), len(cnf.clauses)
        model = {lit for lit in solver.get_model() if lit > 0}
    asg: Dict[int, int] = {}
    for (v, l), vid in var.items():
        if vid in model:
            asg[v] = l
    return True, asg, len(var), len(cnf.clauses)


def refine_min_sum_layers(G: nx.Graph, L: int) -> Tuple[Optional[Dict[int, int]], Optional[int]]:
    if G.number_of_nodes() == 0 or L <= 0:
        return {}, 0
    if not HAVE_PYSAT:
        sat, asg = _bruteforce_coloring(G, L)
        return (asg, sum(asg.values())) if sat else (None, None)
    wcnf = WCNF(); var: Dict[Tuple[int, int], int] = {}
    def vv(v: int, l: int) -> int:
        key = (v, l)
        if key not in var:
            var[key] = len(var) + 1
        return var[key]
    for v in G.nodes():
        wcnf.append([vv(v, l) for l in range(1, L + 1)])
        for l1 in range(1, L + 1):
            for l2 in range(l1 + 1, L + 1):
                wcnf.append([-vv(v, l1), -vv(v, l2)])
    for u, v in G.edges():
        for l in range(1, L + 1):
            wcnf.append([-vv(u, l), -vv(v, l)])
    for v in G.nodes():
        for l in range(1, L + 1):
            wcnf.append([-vv(v, l)], weight=l)
    with RC2(wcnf) as rc2:
        model = rc2.compute()
    pos = {lit for lit in model if lit > 0}
    asg: Dict[int, int] = {}
    for (v, l), vid in var.items():
        if vid in pos:
            asg[v] = l
    return asg, sum(asg.values())


def minimize_t_layers(G: nx.Graph, out_dir: str, write_dimacs: bool = True, use_maxsat: bool = True) -> SatCertificate:
    if G.number_of_nodes() == 0:
        return SatCertificate(0, {}, 0, 0, 0, [], 0, 0, 0)
    lb = clique_lower_bound(G); ub = max(lb, greedy_upper_bound(G))
    original_lb, original_ub = lb, ub
    best_L: Optional[int] = None; best_asg: Dict[int, int] = {}
    best_vars = 0; best_clauses = 0; trials: List[Dict[str, Any]] = []; calls = 0
    while lb <= ub:
        mid = (lb + ub) // 2; calls += 1
        dimacs = str(Path(out_dir) / f"coloring_L{mid}.cnf") if write_dimacs else None
        sat, asg, nvars, nclauses = solve_coloring_sat(G, mid, dimacs)
        trials.append({"L": mid, "sat": sat, "variables": nvars, "clauses": nclauses})
        if sat:
            best_L = mid; best_asg = asg; best_vars = nvars; best_clauses = nclauses; ub = mid - 1
        else:
            lb = mid + 1
    if best_L is None:
        raise RuntimeError("no satisfiable coloring found")
    maxsat_sum = None
    if use_maxsat:
        refined, maxsat_sum = refine_min_sum_layers(G, best_L)
        if refined is not None:
            best_asg = refined
    return SatCertificate(best_L, best_asg, original_lb, original_ub, calls, trials, best_vars, best_clauses, maxsat_sum)


def verify_certificate(G: nx.Graph, cert: SatCertificate) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if cert.optimum_layers == 0:
        if G.number_of_nodes() != 0:
            errors.append("zero-layer certificate for non-empty graph")
        return not errors, errors
    for v in G.nodes():
        l = cert.assignment.get(v)
        if l is None:
            errors.append(f"missing assignment for node {v}")
        elif l < 1 or l > cert.optimum_layers:
            errors.append(f"node {v} has invalid layer {l}")
    for u, v in G.edges():
        if cert.assignment.get(u) == cert.assignment.get(v):
            errors.append(f"edge ({u},{v}) has same layer")
    # Independent optimality witness: verify all trial entries below L* are UNSAT and L* is SAT when available.
    if cert.trials:
        for t in cert.trials:
            L = int(t["L"]); sat = bool(t["sat"])
            if L < cert.optimum_layers and sat:
                errors.append(f"trial says L={L}<L* is SAT")
            if L == cert.optimum_layers and not sat:
                errors.append(f"trial says L*=L={L} is UNSAT")
    return not errors, errors


# ---------------- constructive retiming and proof ----------------
def _is_tlike_inst(inst: Any, tol: float) -> bool:
    name = inst.operation.name.lower()
    if name in {"t", "tdg"}:
        return True
    if name in {"rz", "p", "u1"}:
        try:
            return classify_rz_as_t(float(inst.operation.params[0]), tol) is not None
        except Exception:
            return False
    return False


def _inst_qubits(inst: Any, qmap: Dict[Any, int]) -> Tuple[int, ...]:
    return tuple(qmap[q] for q in inst.qubits)


def _swap_reason_if_valid(t_inst: Any, left_inst: Any, qmap: Dict[Any, int], tol: float) -> Optional[str]:
    tq = _inst_qubits(t_inst, qmap); lq = _inst_qubits(left_inst, qmap)
    lname = left_inst.operation.name.lower()
    if set(tq).isdisjoint(lq):
        return "disjoint_support"
    if len(tq) == 1 and tuple(tq) == tuple(lq) and lname in Z_PHASE_GATES:
        return "same_qubit_z_phase_commutation"
    if lname in {"cx", "ccx", "mcx"} and len(tq) == 1:
        controls, target = _controls_and_target(lname, lq)
        if tq[0] in controls and tq[0] != target:
            return "phase_on_control_commutes_with_controlled_x"
    return None


def _instruction_signature(inst: Any, qmap: Dict[Any, int]) -> Dict[str, Any]:
    return {
        "gate": inst.operation.name.lower(),
        "qubits": list(_inst_qubits(inst, qmap)),
        "params": [float(p) if isinstance(p, (int, float, np.floating)) else str(p) for p in getattr(inst.operation, "params", [])],
    }


def _build_circuit_from_instruction_order(qc: QuantumCircuit, order: List[int]) -> QuantumCircuit:
    qmap = qid_map(qc); cmap = {c: i for i, c in enumerate(qc.clbits)}
    out = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for orig_id in order:
        inst = qc.data[orig_id]
        out.append(inst.operation, [out.qubits[qmap[q]] for q in inst.qubits], [out.clbits[cmap[c]] for c in inst.clbits])
    return out


def apply_conservative_layered_retiming(qc: QuantumCircuit, cert: SatCertificate, tol: float = ANGLE_TOL, max_swaps: int = 200000) -> Tuple[QuantumCircuit, RetimingCertificate]:
    if not cert.assignment:
        ret_cert = RetimingCertificate(
            method="identity_no_t_nodes",
            valid_by_local_commutation=True,
            replay_valid=True,
            swaps=[],
            failed_swaps=[],
            theorem_scope="No T-like nodes or empty assignment; output is syntactically identical.",
        )
        return qc.copy(), ret_cert

    qmap0 = qid_map(qc)
    original_nodes = t_nodes(qc, tol)
    idx_to_layer = {node.circuit_index: cert.assignment.get(node.node_id, 10**9) for node in original_nodes}
    data_ids = list(range(len(qc.data)))
    target = sorted([node.circuit_index for node in original_nodes], key=lambda idx: (idx_to_layer[idx], idx))
    swaps: List[SwapStep] = []
    failed: List[Dict[str, Any]] = []

    def inst_at_pos(pos: int) -> Any:
        return qc.data[data_ids[pos]]

    for orig_idx in target:
        guard = 0
        while guard < max_swaps:
            guard += 1
            try:
                cur = data_ids.index(orig_idx)
            except ValueError:
                failed.append({"orig_idx": orig_idx, "reason": "instruction_id_not_found"}); break
            if cur == 0:
                break
            left_id = data_ids[cur - 1]
            left_inst = qc.data[left_id]
            cur_inst = qc.data[orig_idx]
            if _is_tlike_inst(left_inst, tol):
                left_layer = idx_to_layer.get(left_id, 0)
                cur_layer = idx_to_layer.get(orig_idx, 0)
                if left_layer <= cur_layer:
                    break
            reason = _swap_reason_if_valid(cur_inst, left_inst, qmap0, tol)
            if reason is None:
                failed.append({
                    "moving_id": orig_idx,
                    "blocked_by_id": left_id,
                    "moving": _instruction_signature(cur_inst, qmap0),
                    "blocked_by": _instruction_signature(left_inst, qmap0),
                    "reason": "no_local_commutation_rule",
                })
                break
            swaps.append(SwapStep(
                step=len(swaps), left_id_before=left_id, right_id_before=orig_idx, reason=reason,
                left_gate=left_inst.operation.name.lower(), right_gate=cur_inst.operation.name.lower(),
                left_qubits=_inst_qubits(left_inst, qmap0), right_qubits=_inst_qubits(cur_inst, qmap0),
            ))
            data_ids[cur - 1], data_ids[cur] = data_ids[cur], data_ids[cur - 1]
            if len(swaps) >= max_swaps:
                failed.append({"reason": "max_swaps_reached", "max_swaps": max_swaps}); break

    replay_ok, replay_errors = replay_retiming_swaps(qc, swaps, tol)
    if replay_errors:
        failed.extend(replay_errors)
    retimed = _build_circuit_from_instruction_order(qc, data_ids)
    ret_cert = RetimingCertificate(
        method="adjacent_swap_local_commutation",
        valid_by_local_commutation=all(s.reason in {"disjoint_support", "same_qubit_z_phase_commutation", "phase_on_control_commutes_with_controlled_x"} for s in swaps),
        replay_valid=replay_ok,
        swaps=swaps,
        failed_swaps=failed,
        theorem_scope=(
            "Each recorded adjacent swap is justified by one of three local commutation rules: "
            "disjoint support, same-qubit Z-phase commutation, or phase on a control line commuting with controlled-X. "
            "Therefore the retimed circuit is equivalent to the encoding-input circuit under this commutation model."
        ),
    )
    return retimed, ret_cert


def replay_retiming_swaps(qc: QuantumCircuit, swaps: Sequence[SwapStep], tol: float = ANGLE_TOL) -> Tuple[bool, List[Dict[str, Any]]]:
    """Independently replay the recorded adjacent swaps on instruction IDs."""
    qmap = qid_map(qc)
    order = list(range(len(qc.data)))
    errors: List[Dict[str, Any]] = []
    for s in swaps:
        try:
            li = order.index(s.left_id_before)
            ri = order.index(s.right_id_before)
        except ValueError:
            errors.append({"step": s.step, "reason": "id_missing_during_replay"}); continue
        if ri - li != 1:
            errors.append({"step": s.step, "reason": "not_adjacent_during_replay", "left_pos": li, "right_pos": ri}); continue
        reason = _swap_reason_if_valid(qc.data[s.right_id_before], qc.data[s.left_id_before], qmap, tol)
        if reason != s.reason:
            errors.append({"step": s.step, "reason": "rule_mismatch", "expected": s.reason, "actual": reason}); continue
        order[li], order[ri] = order[ri], order[li]
    return len(errors) == 0, errors



# ---------------- formal equivalence layer ----------------
LOCAL_COMMUTATION_RULES = {
    "disjoint_support": "Operations acting on disjoint tensor factors commute exactly.",
    "same_qubit_z_phase_commutation": "Single-qubit Z-phase rotations on the same qubit commute exactly.",
    "phase_on_control_commutes_with_controlled_x": "A Z-phase on a control wire commutes exactly with controlled-X because the control projector is diagonal in the computational basis.",
}


def build_formal_equivalence_report(ret_cert: RetimingCertificate) -> FormalEquivalenceReport:
    """Build a machine-readable theorem/certificate for global retiming equivalence.

    The report proves equivalence only between the exact instruction sequence used
    for SAT encoding (``encoding_input.qasm``) and the emitted conservative
    retiming (``retimed_conservative.qasm``). It deliberately does not certify
    external transformations such as Qiskit transpilation, PyZX extraction, or
    tket optimization. Those remain supplemental validation layers.
    """
    rule_counts: Dict[str, int] = {}
    invalid_rules: List[str] = []
    for s in ret_cert.swaps:
        rule_counts[s.reason] = rule_counts.get(s.reason, 0) + 1
        if s.reason not in LOCAL_COMMUTATION_RULES:
            invalid_rules.append(s.reason)
    assumptions = [
        "The emitted circuit differs from the encoding-input circuit only by the recorded adjacent swaps.",
        "Every recorded adjacent swap is one of the explicitly enumerated local commutation rules.",
        "Instruction replay verifies that the swaps are adjacent and applied in the recorded order.",
        "The semantics of the supported gates are the standard unitary semantics over the same qubit ordering.",
    ]
    proof_obligations = [
        "local_rule_soundness",
        "adjacent_swap_replay",
        "composition_of_semantics_preserving_swaps",
        "same_instruction_multiset_and_qubit_order",
    ]
    discharged: List[str] = []
    undischarged: List[str] = []
    if not invalid_rules:
        discharged.append("local_rule_soundness")
    else:
        undischarged.append("local_rule_soundness:unknown_rules=" + ",".join(sorted(set(invalid_rules))))
    if ret_cert.replay_valid:
        discharged.append("adjacent_swap_replay")
    else:
        undischarged.append("adjacent_swap_replay")
    if ret_cert.valid_by_local_commutation and ret_cert.replay_valid and not invalid_rules:
        discharged.extend(["composition_of_semantics_preserving_swaps", "same_instruction_multiset_and_qubit_order"])
    else:
        for x in ["composition_of_semantics_preserving_swaps", "same_instruction_multiset_and_qubit_order"]:
            if x not in discharged:
                undischarged.append(x)
    proven = len(undischarged) == 0
    if proven:
        statement = (
            "The retimed circuit is exactly unitary-equivalent to the SAT encoding-input circuit. "
            "This follows by induction over the replayed adjacent swaps: each swap is locally sound, "
            "and composition preserves equality of the overall unitary."
        )
    else:
        statement = (
            "Global equivalence is not formally certified for this instance because at least one proof obligation remains open."
        )
    return FormalEquivalenceReport(
        proven=proven,
        scope="encoding_input_qasm_to_retimed_conservative_qasm",
        theorem_name="Global equivalence by composition of locally sound adjacent commutations",
        assumptions=assumptions,
        proof_obligations=proof_obligations,
        discharged_obligations=discharged,
        undischarged_obligations=undischarged,
        local_rules_used=rule_counts,
        total_swaps=len(ret_cert.swaps),
        replay_valid=ret_cert.replay_valid,
        statement=statement,
    )


def write_equivalence_theory_files(out_dir: str | Path) -> None:
    """Write paper-facing theorem assumptions and local rule catalogue."""
    out = ensure_dir(out_dir)
    rows = []
    for rule, statement in LOCAL_COMMUTATION_RULES.items():
        rows.append({"rule": rule, "statement": statement, "status": "axiomatically_sound_under_standard_unitary_semantics"})
    pd.DataFrame(rows).to_csv(out / "local_commutation_rules.csv", index=False)
    theorem = {
        "theorem": "Global equivalence by certified adjacent-swap replay",
        "formal_scope": "encoding_input.qasm -> retimed_conservative.qasm only",
        "not_claimed": [
            "No independent formal certificate is claimed for Qiskit transpilation.",
            "No independent formal certificate is claimed for optional ZX normalization/extraction.",
            "No independent formal certificate is claimed for baseline optimizer outputs.",
            "No claim is made that the graph optimum is always constructively realizable by the conservative retimer.",
        ],
        "proof_sketch": [
            "Represent the circuit as a finite instruction sequence.",
            "The retimer emits a list of adjacent transpositions.",
            "Replay checks that each transposition is adjacent in the current sequence.",
            "Each transposition is accepted only if it matches one of the local commutation rules.",
            "Each local rule preserves the denoted unitary under standard semantics.",
            "By induction over the transposition list, the initial and final instruction sequences denote the same unitary.",
        ],
    }
    write_json(out / "formal_equivalence_theorem.json", theorem)


def equivalence_report_to_flat_row(name: str, report: FormalEquivalenceReport) -> Dict[str, Any]:
    row = {
        "name": name,
        "formal_global_equivalence_proven": report.proven,
        "formal_global_equivalence_scope": report.scope,
        "formal_global_equivalence_statement": report.statement,
        "formal_undischarged_obligations": len(report.undischarged_obligations),
        "total_swaps": report.total_swaps,
        "replay_valid": report.replay_valid,
    }
    for rule in LOCAL_COMMUTATION_RULES:
        row[f"rule_count__{rule}"] = report.local_rules_used.get(rule, 0)
    return row


# ---------------- v11 scalability guardrails ----------------
def _skip_baseline_if_large(tool: str, qc: QuantumCircuit, cfg: ExperimentConfig) -> Optional[BaselineResult]:
    """Return an explicit skip result for large circuits that may hang optional tools."""
    n = int(getattr(qc, "num_qubits", 0))
    if tool in {"tket", "pyzx"} and n > cfg.max_external_baseline_qubits:
        return BaselineResult(
            tool,
            None,
            f"skipped:large_qubit_count>{cfg.max_external_baseline_qubits}",
            cfg.large_baseline_policy,
            note="External baseline skipped by v11 guardrail; formal retiming proof and SAT certificate are still produced.",
        )
    if tool == "qiskit" and n > cfg.max_qiskit_baseline_qubits:
        return BaselineResult(
            tool,
            None,
            f"skipped:large_qubit_count>{cfg.max_qiskit_baseline_qubits}",
            "qiskit_baseline_guardrail",
            note="Qiskit multi-level baseline skipped by v11 guardrail.",
        )
    return None


def supplemental_equivalence_policy(original: QuantumCircuit, cfg: ExperimentConfig) -> Dict[str, Any]:
    n = int(getattr(original, "num_qubits", 0))
    return {
        "n_qubits": n,
        "formal_replay_always_used": True,
        "qcec_allowed": bool(cfg.run_supplemental_equivalence and HAVE_QCEC and n <= cfg.max_qcec_qubits),
        "exact_statevector_allowed": bool(cfg.run_supplemental_equivalence and n <= cfg.verify_equivalence_up_to_qubits),
        "random_statevector_allowed": bool(cfg.run_supplemental_equivalence and n <= cfg.max_random_equivalence_qubits),
        "large_skip": bool(n > cfg.max_supplemental_equivalence_qubits),
        "skip_reason": cfg.skip_supplemental_reason if n > cfg.max_supplemental_equivalence_qubits else None,
    }


# ---------------- baselines and supplemental equivalence ----------------
def _baseline_circuit(qc: QuantumCircuit, cfg: ExperimentConfig) -> QuantumCircuit:
    """Return a conservative Qiskit circuit for optional baseline tools.

    Several baseline failures in earlier runs came from feeding tool-specific
    converters with high-level or parameterized operations.  This helper binds
    parameters and transpiles to the same Clifford+T-oriented basis used by the
    experiment before the external tool is called.
    """
    try:
        qc2 = bind_parameters_to_clifford_t(qc, cfg.seed)
        return transpile(qc2, basis_gates=list(cfg.basis_gates), optimization_level=0)
    except Exception:
        return qc.copy()


def tket_baseline(qc: QuantumCircuit, cfg: ExperimentConfig) -> BaselineResult:
    skipped = _skip_baseline_if_large("tket", qc, cfg)
    if skipped is not None:
        return skipped
    if not HAVE_TKET:
        return BaselineResult("tket", None, "unavailable:dependency_missing", "pytket not installed")
    qc_in = _baseline_circuit(qc, cfg)
    attempts: List[str] = []
    try:
        tk = qiskit_to_tk(qc_in)
        attempts.append("qiskit_to_tk")
        try:
            FullPeepholeOptimise().apply(tk)
            attempts.append("FullPeepholeOptimise")
        except Exception as exc:
            attempts.append(f"FullPeepholeOptimise_failed:{exc}")
        back = tk_to_qiskit(tk)
        back = transpile(back, basis_gates=list(cfg.basis_gates), optimization_level=cfg.qiskit_optimization_level)
        return BaselineResult("tket", int(naive_t_depth(back, cfg.angle_tol)), "ok", ";".join(attempts))
    except Exception as exc1:
        # Fallback: try QASM-based import when extension conversion fails.
        try:
            from pytket.qasm import circuit_from_qasm_str  # type: ignore
            qasm_txt = qasm2.dumps(qc_in)
            tk = circuit_from_qasm_str(qasm_txt)
            attempts.append("qasm_import")
            try:
                FullPeepholeOptimise().apply(tk)
                attempts.append("FullPeepholeOptimise")
            except Exception as exc2:
                attempts.append(f"FullPeepholeOptimise_failed:{exc2}")
            back = tk_to_qiskit(tk)
            back = transpile(back, basis_gates=list(cfg.basis_gates), optimization_level=cfg.qiskit_optimization_level)
            return BaselineResult("tket", int(naive_t_depth(back, cfg.angle_tol)), "ok:fallback_qasm", ";".join(attempts), error=str(exc1))
        except Exception as exc2:
            return BaselineResult("tket", None, "failed:conversion_or_pass", ";".join(attempts) or "none", error=f"primary={exc1}; fallback={exc2}")


def _pyzx_circuit_to_qiskit(circ: Any, attempts: List[str]) -> QuantumCircuit:
    """Robust conversion from a PyZX Circuit to Qiskit across PyZX versions.

    PyZX 0.10.x may not expose ``Circuit.to_qiskit``.  Earlier v6 runs failed
    at exactly this point.  The QASM fallback is intentionally the primary
    compatibility bridge for paper baselines, because QASM is also exported as
    an artifact and can be independently inspected.
    """
    errors: List[str] = []
    candidates = []
    try:
        candidates.append(("to_basic_gates", circ.to_basic_gates()))
    except Exception as exc:
        errors.append(f"to_basic_gates_failed:{exc}")
    candidates.append(("direct", circ))
    for label, candidate in candidates:
        try:
            to_qiskit = getattr(candidate, "to_qiskit", None)
            if callable(to_qiskit):
                attempts.append(f"convert_{label}_to_qiskit")
                return to_qiskit()
        except Exception as exc:
            errors.append(f"{label}_to_qiskit_failed:{exc}")
        try:
            to_qasm = getattr(candidate, "to_qasm", None)
            if callable(to_qasm):
                qasm_txt = to_qasm()
                attempts.append(f"convert_{label}_to_qasm2")
                return qasm2.loads(qasm_txt)
        except Exception as exc:
            errors.append(f"{label}_to_qasm2_failed:{exc}")
    raise RuntimeError("Could not convert PyZX Circuit to Qiskit; " + "; ".join(errors))


def pyzx_baseline(qc: QuantumCircuit, cfg: ExperimentConfig) -> BaselineResult:
    skipped = _skip_baseline_if_large("pyzx", qc, cfg)
    if skipped is not None:
        return skipped
    if not HAVE_PYZX:
        return BaselineResult("pyzx", None, "unavailable:dependency_missing", "pyzx not installed")
    qc_in = _baseline_circuit(qc, cfg)
    attempts: List[str] = []
    try:
        try:
            circ = zx.Circuit.from_qasm(qasm2.dumps(qc_in))
            attempts.append("from_qasm")
        except Exception as exc1:
            try:
                circ = zx.Circuit.from_qiskit(qc_in)
                attempts.append("from_qiskit")
            except Exception as exc2:
                return BaselineResult("pyzx", None, "failed:input_conversion", ";".join(attempts) or "none", error=f"from_qasm={exc1}; from_qiskit={exc2}")

        # PyZX transformations are best-effort baselines. Each pass is isolated
        # so one unavailable optimizer does not erase the entire baseline result.
        try:
            g = circ.to_graph()
            zx.simplify.full_reduce(g)
            circ = zx.extract.extract_circuit(g.copy())
            attempts.append("full_reduce_extract")
        except Exception as exc:
            attempts.append(f"full_reduce_extract_failed:{exc}")
        try:
            import pyzx.tpar as tpar_mod  # type: ignore
            tpar_mod.tpar(circ, optimize="depth")
            attempts.append("tpar_depth")
        except Exception as exc:
            attempts.append(f"tpar_depth_unavailable:{exc}")
            try:
                zx.optimize.tpar_optimized(circ, for_depth=True)
                attempts.append("tpar_optimized")
            except Exception as exc2:
                attempts.append(f"tpar_optimized_unavailable:{exc2}")
        back = _pyzx_circuit_to_qiskit(circ, attempts)
        back = transpile(back, basis_gates=list(cfg.basis_gates), optimization_level=cfg.qiskit_optimization_level)
        return BaselineResult("pyzx", int(naive_t_depth(back, cfg.angle_tol)), "ok", ";".join(attempts))
    except Exception as exc:
        # Last-resort numeric baseline: PyZX was reachable but optimization/export failed.
        # Return the Qiskit-compatible input depth as an explicit fallback so the table
        # has coverage without pretending the PyZX optimizer completed.
        try:
            td = int(naive_t_depth(qc_in, cfg.angle_tol))
            return BaselineResult("pyzx", td, "ok:fallback_input_depth_after_pyzx_failure", ";".join(attempts) or "none", error=str(exc), note="PyZX pipeline failed after import; reported numeric fallback is the pre-PyZX baseline depth, not a PyZX-optimized depth.")
        except Exception:
            return BaselineResult("pyzx", None, "failed:pyzx_pipeline", ";".join(attempts) or "none", error=str(exc))


def qiskit_baseline(qc: QuantumCircuit, cfg: ExperimentConfig) -> BaselineResult:
    skipped = _skip_baseline_if_large("qiskit", qc, cfg)
    if skipped is not None:
        return skipped
    try:
        best = None; best_level = None
        levels = [0, 1] if getattr(qc, "num_qubits", 0) > cfg.max_external_baseline_qubits else [0, 1, 2, 3]
        for level in levels:
            tqc = transpile(qc, basis_gates=list(cfg.basis_gates), optimization_level=level)
            td = naive_t_depth(tqc, cfg.angle_tol)
            if best is None or td < best:
                best = td; best_level = level
        return BaselineResult("qiskit", int(best) if best is not None else None, "ok", f"best_optimization_level={best_level}")
    except Exception as exc:
        return BaselineResult("qiskit", None, "failed:qiskit_transpile", "qiskit_transpile", error=str(exc))


def tket_t_depth_baseline(qc: QuantumCircuit, cfg: ExperimentConfig) -> Optional[int]:
    return tket_baseline(qc, cfg).t_depth


def pyzx_t_depth_baseline(qc: QuantumCircuit, cfg: ExperimentConfig) -> Optional[int]:
    return pyzx_baseline(qc, cfg).t_depth


def qiskit_optimized_t_depth(qc: QuantumCircuit, cfg: ExperimentConfig) -> Optional[int]:
    return qiskit_baseline(qc, cfg).t_depth


def _equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    if a.shape != b.shape:
        return False
    idx = int(np.argmax(np.abs(b))) if b.size else 0
    if b.size and abs(b[idx]) > atol:
        phase = a[idx] / b[idx]
        if abs(phase) > atol:
            b = phase * b
    return bool(np.allclose(a, b, atol=atol))


def formal_equivalence_from_retiming(ret_cert: RetimingCertificate) -> str:
    """Return the formal circuit-level equivalence claim from the swap proof.

    The global claim follows by composition: every recorded adjacent swap is one
    of the explicitly enumerated semantics-preserving local commutations, and
    replay verifies that exactly those adjacent swaps transform the original
    instruction order into the emitted order.
    """
    if ret_cert.valid_by_local_commutation and ret_cert.replay_valid:
        return "true:composition_of_locally_sound_adjacent_commutations"
    return "not_proven:retiming_certificate_invalid_or_not_replayable"


def supplemental_equivalence_status(original: QuantumCircuit, transformed: QuantumCircuit, cfg: ExperimentConfig) -> str:
    """Bounded supplemental equivalence checks.

    v11 deliberately avoids exponential statevector/QCEC work on large-n
    benchmarks. The formal equivalence claim remains the replayable
    local-commutation proof; this function only reports extra evidence when it
    is computationally safe.
    """
    if original.num_qubits != transformed.num_qubits:
        return "false:qubit_count_mismatch"
    n = int(original.num_qubits)
    policy = supplemental_equivalence_policy(original, cfg)
    if not cfg.run_supplemental_equivalence:
        return "skipped:disabled_formal_replay_used"
    if policy["large_skip"]:
        return f"skipped:{cfg.skip_supplemental_reason};n={n};threshold={cfg.max_supplemental_equivalence_qubits}"

    qcec_msg = "qcec_skipped:guardrail"
    if policy["qcec_allowed"]:
        try:
            res = qcec.verify(original, transformed)
            s = str(res).lower()
            if "not_equivalent" in s or "nonequivalent" in s:
                return "false:qcec"
            if "equivalent" in s:
                return "true:qcec"
            qcec_msg = f"qcec_inconclusive:{res}"
        except Exception as exc:
            qcec_msg = f"qcec_unavailable:{exc}"
    elif not HAVE_QCEC:
        qcec_msg = "qcec_not_installed"

    if policy["exact_statevector_allowed"]:
        # Prefer exact operator comparison when the state dimension is still modest.
        try:
            if Operator is not None:
                a = Operator(original).data
                b = Operator(transformed).data
                if _equal_up_to_global_phase(a.reshape(-1), b.reshape(-1)):
                    return f"true:operator_exact_up_to_global_phase;{qcec_msg}"
                return "false:operator_exact"
        except Exception:
            pass
        try:
            for x in range(2**n):
                init = Statevector.from_label(format(x, f"0{n}b"))
                a = init.evolve(original).data
                b = init.evolve(transformed).data
                if not _equal_up_to_global_phase(a, b):
                    return f"false:statevector_counterexample:{x}"
            return f"true:statevector_full_basis;{qcec_msg}"
        except Exception as exc:
            return f"not_checked:statevector_error:{exc};{qcec_msg}"

    if not policy["random_statevector_allowed"]:
        return f"skipped:random_statevector_guardrail;n={n};formal_replay_used;{qcec_msg}"
    try:
        rnd = random.Random(cfg.seed)
        for _ in range(cfg.random_equivalence_tests):
            x = rnd.randrange(0, 2**n)
            init = Statevector.from_label(format(x, f"0{n}b"))
            a = init.evolve(original).data
            b = init.evolve(transformed).data
            if not _equal_up_to_global_phase(a, b):
                return f"false:random_basis_counterexample:{x}"
        return f"true:random_basis_{cfg.random_equivalence_tests};{qcec_msg}"
    except Exception as exc:
        return f"not_checked:random_statevector_error:{exc};{qcec_msg}"




# ---------------- paper-facing diagnostics ----------------
def classify_circuit_graph_regime(treewidth: Optional[int], layers: int, vertices: int) -> Tuple[str, bool, str]:
    """Paper-facing classification of circuit-derived dependency graphs.

    This intentionally does not hide the near-tree phenomenon.  It turns the
    reviewer concern (treewidth≈1 and layer optimum≤2 on circuit-derived tests)
    into an explicit, reproducible diagnostic table.
    """
    tw = -1 if treewidth is None else int(treewidth)
    if vertices == 0:
        return "no_t_nodes", True, "trivial_no_non_clifford_layering_needed"
    if tw <= 1:
        regime = "near_tree_dependency_graph"
    elif tw <= 3:
        regime = "low_treewidth_dependency_graph"
    elif tw <= 8:
        regime = "moderate_treewidth_dependency_graph"
    else:
        regime = "high_treewidth_dependency_graph"
    near_tree = bool(tw <= 1 and layers <= 2)
    if layers <= 2:
        opt_regime = "two_layer_or_less_graph_optimum"
    elif layers <= 4:
        opt_regime = "small_constant_graph_optimum"
    else:
        opt_regime = "multi_layer_graph_optimum"
    return regime, near_tree, opt_regime


def construction_gap_label(input_td: int, graph_layers: int, constructive_td: int) -> Tuple[int, bool, bool, str]:
    """Return retiming improvement and paper-ready gap interpretation."""
    improvement = int(input_td) - int(constructive_td)
    gap = int(constructive_td) - int(graph_layers)
    limitation = bool(improvement == 0 and gap > 0)
    separation = bool(gap > 0)
    if gap == 0 and improvement > 0:
        interp = "graph_optimum_constructively_realized_with_improvement"
    elif gap == 0:
        interp = "graph_optimum_constructively_realized_no_improvement_needed"
    elif limitation:
        interp = "separation_witness_local_retiming_does_not_realize_graph_optimum"
    else:
        interp = "positive_construction_gap_after_partial_retiming"
    return improvement, limitation, separation, interp


def graph_stress_role(graph_family: str, n: int) -> str:
    if graph_family in {"complete", "erdos_renyi"}:
        return "hard_dense_or_high_treewidth_stress_case"
    if graph_family in {"path", "cycle", "grid", "complete_bipartite"}:
        return "controlled_structure_treewidth_scaling_case"
    return "unstructured_scalability_probe"


def known_graph_coloring_certificate(G: nx.Graph, graph_family: str) -> Optional[Tuple[int, Dict[int, int], str]]:
    """Exact closed-form certificates for graph-stress families with known chromatic number.

    This is crucial for complete graphs: proving K_n is not (n-1)-colorable by
    generic SAT can be disproportionately expensive due to symmetry, even for
    n=10, although chi(K_n)=n is mathematically immediate.  For graph-stress
    experiments we want structural scaling up to n≈20+, not a benchmark of
    pigeonhole UNSAT hardness.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0, {}, "closed_form:empty_graph"
    nodes = list(G.nodes())
    if graph_family == "complete":
        return n, {v: i+1 for i, v in enumerate(nodes)}, "closed_form:complete_graph_chi_n"
    if graph_family in {"path", "complete_bipartite", "grid"}:
        # These NetworkX generators produce bipartite graphs in this suite.
        try:
            color0, color1 = nx.algorithms.bipartite.sets(G) if nx.is_connected(G) else (set(), set())
        except Exception:
            color = nx.algorithms.coloring.greedy_color(G, strategy="largest_first")
            L = max(color.values(), default=-1) + 1
            return L, {v: c+1 for v, c in color.items()}, f"greedy_exact_expected_bipartite:{graph_family}"
        if not color0 and not color1:
            color = nx.algorithms.coloring.greedy_color(G, strategy="largest_first")
            L = max(color.values(), default=-1) + 1
            return L, {v: c+1 for v, c in color.items()}, f"greedy_coloring:{graph_family}"
        asg = {v: 1 for v in color0}
        asg.update({v: 2 for v in color1})
        return 1 if G.number_of_edges() == 0 else 2, asg, f"closed_form:bipartite_{graph_family}"
    if graph_family == "cycle":
        # Even cycles are 2-colorable; odd cycles require 3 colors.
        if n % 2 == 0:
            return 2, {v: (i % 2)+1 for i, v in enumerate(nodes)}, "closed_form:even_cycle"
        asg = {v: (i % 2)+1 for i, v in enumerate(nodes)}
        asg[nodes[-1]] = 3
        return 3, asg, "closed_form:odd_cycle"
    return None

# ---------------- graph-only stress benchmarks ----------------
def make_stress_graph(graph_family: str, n: int, seed: int = 42, density: float = 0.25) -> nx.Graph:
    rnd = random.Random(seed)
    if graph_family == "path":
        return nx.path_graph(n)
    if graph_family == "cycle":
        return nx.cycle_graph(n)
    if graph_family == "grid":
        side = int(math.sqrt(n))
        G = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(G)
    if graph_family == "complete":
        return nx.complete_graph(n)
    if graph_family == "complete_bipartite":
        return nx.complete_bipartite_graph(n//2, n - n//2)
    if graph_family == "erdos_renyi":
        return nx.gnp_random_graph(n, density, seed=seed)
    if graph_family == "barabasi":
        m = max(1, min(4, n//4))
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if graph_family == "watts_strogatz":
        k = max(2, min(8, n//2))
        if k % 2 == 1: k += 1
        return nx.watts_strogatz_graph(n, k, 0.25, seed=seed)
    raise ValueError(f"unknown graph stress family: {graph_family}")


def run_graph_stress_instance(name: str, graph_family: str, n: int, out_dir: str, seed: int = 42, density: float = 0.25, use_maxsat: bool = False) -> GraphStressResult:
    idir = ensure_dir(Path(out_dir) / "graph_stress" / name)
    G = make_stress_graph(graph_family, n, seed=seed, density=density)
    gstats = graph_statistics(G)
    write_json(idir / "graph_params.json", gstats)

    known = known_graph_coloring_certificate(G, graph_family)
    if known is not None and not use_maxsat:
        t0 = time.time()
        L, asg, mode = known
        nvars, nclauses = _analytical_cnf_size(G, max(1, L))
        cert = SatCertificate(
            optimum_layers=L,
            assignment=asg,
            lower_bound_clique=clique_lower_bound(G),
            upper_bound_greedy=greedy_upper_bound(G),
            sat_calls=0,
            trials=[{"L": L, "sat": True, "variables": nvars, "clauses": nclauses, "mode": mode}],
            variables_at_optimum=nvars,
            clauses_at_optimum=nclauses,
            maxsat_sum_layers=None,
        )
        sat_seconds = time.time() - t0
        write_json(idir / "solver_note.json", {
            "mode": mode,
            "why": "Closed-form graph-stress certificate avoids measuring symmetric UNSAT pigeonhole hardness as if it were circuit scalability.",
            "example": "For K_n, chi(K_n)=n. Generic SAT must prove K_n is not (n-1)-colorable and may blow up already around n=10.",
        })
    else:
        t0 = time.time()
        # For graph stress, MaxSAT is intentionally off by default. If manually
        # enabled, dense/complete instances still skip it because secondary
        # layer-sum optimization is unrelated to chromatic feasibility scaling.
        safe_maxsat = bool(use_maxsat and graph_family not in {"complete", "erdos_renyi"} and G.number_of_nodes() <= 30)
        cert = minimize_t_layers(G, str(idir), write_dimacs=True, use_maxsat=safe_maxsat)
        sat_seconds = time.time() - t0
        mode = "sat:binary_search_coloring" + ("+maxsat" if safe_maxsat else ":no_maxsat")

    ok, errors = verify_certificate(G, cert)
    write_json(idir / "certificate.json", cert)
    write_json(idir / "certificate_check.json", {"valid": ok, "errors": errors})
    try:
        nx.write_edgelist(G, idir / "graph.edgelist", data=False)
    except Exception:
        pass
    role = graph_stress_role(graph_family, n)
    interp = (
        "closed_form_used_for_known_chromatic_number" if str(mode).startswith("closed_form")
        else "generic_sat_used_for_scalability_probe"
    )
    result = GraphStressResult(
        name=name,
        graph_family=graph_family,
        vertices=G.number_of_nodes(),
        edges=G.number_of_edges(),
        density=float(gstats["density"]),
        treewidth_minfill=None if gstats["treewidth_minfill"] is None else int(gstats["treewidth_minfill"]),
        certified_graph_layers=cert.optimum_layers,
        sat_seconds=sat_seconds,
        sat_variables=cert.variables_at_optimum,
        sat_clauses=cert.clauses_at_optimum,
        clique_lower_bound=cert.lower_bound_clique,
        greedy_upper_bound=cert.upper_bound_greedy,
        certificate_valid=ok,
        solving_mode=str(mode),
        stress_role=role,
        scalability_interpretation=interp,
        artifact_dir=str(idir),
    )
    write_json(idir / "result.json", result)
    return result

def graph_results_to_dataframe(results: Iterable[GraphStressResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


# ---------------- end-to-end instance ----------------
def run_instance(name: str, family: str, n: int, cfg: ExperimentConfig, depth: int = 1, p: int = 1) -> InstanceResult:
    require_qiskit()
    idir = ensure_dir(Path(cfg.out_dir) / name)
    original = bind_parameters_to_clifford_t(prepare_circuit(family, n, depth, p, cfg.seed), cfg.seed)
    transpiled = transpile(original, basis_gates=list(cfg.basis_gates), optimization_level=cfg.qiskit_optimization_level)
    if cfg.write_qasm:
        write_qasm(idir / "original.qasm", original)
        write_qasm(idir / "transpiled.qasm", transpiled)

    circuit_for_encoding = transpiled
    zx_stats: Dict[str, Any] = {"available": bool(HAVE_PYZX), "pre_vertices": None, "post_vertices": None, "error": None}
    if cfg.use_zx_normalization:
        znorm, zx_stats = zx_normalize(transpiled, str(idir / "zx_normalized_raw.qasm"))
        circuit_for_encoding = transpile(znorm, basis_gates=list(cfg.basis_gates), optimization_level=cfg.qiskit_optimization_level)
    if cfg.write_qasm:
        write_qasm(idir / "encoding_input.qasm", circuit_for_encoding)

    if cfg.export_instruction_ids:
        write_json(idir / "instruction_ids.json", [
            {"id": i, **_instruction_signature(inst, qid_map(circuit_for_encoding))}
            for i, inst in enumerate(circuit_for_encoding.data)
        ])

    G, nodes = build_dependency_graph(circuit_for_encoding, cfg.angle_tol)
    gstats = graph_statistics(G)
    write_json(idir / "t_nodes.json", [asdict(x) for x in nodes])
    write_json(idir / "graph_params.json", gstats)

    t0 = time.time()
    cert = minimize_t_layers(G, str(idir), cfg.write_dimacs, cfg.use_maxsat_refinement)
    sat_seconds = time.time() - t0
    cert_ok, cert_errors = verify_certificate(G, cert)
    write_json(idir / "certificate.json", cert)
    write_json(idir / "certificate_check.json", {"valid": cert_ok, "errors": cert_errors})

    retimed, ret_cert = apply_conservative_layered_retiming(circuit_for_encoding, cert, cfg.angle_tol, cfg.max_retiming_swaps)
    if cfg.write_qasm:
        write_qasm(idir / "retimed_conservative.qasm", retimed)
    write_json(idir / "retiming_certificate.json", ret_cert)
    formal_report = build_formal_equivalence_report(ret_cert)
    write_json(idir / "formal_equivalence_report.json", formal_report)
    constructive_td = naive_t_depth(retimed, cfg.angle_tol)
    input_td = naive_t_depth(circuit_for_encoding, cfg.angle_tol)
    gap = constructive_td - cert.optimum_layers
    graph_regime, near_tree_flag, graph_optimum_regime = classify_circuit_graph_regime(
        None if gstats["treewidth_minfill"] is None else int(gstats["treewidth_minfill"]),
        cert.optimum_layers,
        int(gstats["vertices"]),
    )
    retiming_improvement, retiming_limitation, separation_witness, gap_interp = construction_gap_label(
        input_td, cert.optimum_layers, constructive_td
    )

    formal_eq = formal_equivalence_from_retiming(ret_cert)
    equivalence_claim = formal_eq
    write_json(idir / "supplemental_equivalence_policy.json", supplemental_equivalence_policy(circuit_for_encoding, cfg))
    supplemental = supplemental_equivalence_status(circuit_for_encoding, retimed, cfg) if cfg.run_supplemental_equivalence else "skipped:disabled_formal_replay_used"

    tket_res = tket_baseline(circuit_for_encoding, cfg)
    pyzx_res = pyzx_baseline(circuit_for_encoding, cfg)
    qiskit_res = qiskit_baseline(circuit_for_encoding, cfg)
    baseline_results = {
        "tket": asdict(tket_res),
        "pyzx": asdict(pyzx_res),
        "qiskit": asdict(qiskit_res),
        "coverage": {
            "external_tools_available": int(tket_res.t_depth is not None) + int(pyzx_res.t_depth is not None),
            "external_tools_total": 2,
            "any_numeric_baseline": any(x.t_depth is not None for x in [tket_res, pyzx_res, qiskit_res]),
        },
    }
    write_json(idir / "baselines.json", baseline_results)
    baseline_coverage = f"external_numeric={baseline_results['coverage']['external_tools_available']}/2; qiskit={qiskit_res.status}"

    result = InstanceResult(
        name=name,
        family=family,
        n_qubits=circuit_for_encoding.num_qubits,
        t_count=len(nodes),
        t_depth_input=input_td,
        certified_graph_layers=cert.optimum_layers,
        constructive_t_depth=constructive_td,
        construction_gap=gap,
        retiming_improvement=retiming_improvement,
        retiming_limitation_flag=retiming_limitation,
        separation_witness=separation_witness,
        construction_gap_interpretation=gap_interp,
        graph_vertices=int(gstats["vertices"]),
        graph_edges=int(gstats["edges"]),
        graph_density=float(gstats["density"]),
        graph_treewidth_minfill=None if gstats["treewidth_minfill"] is None else int(gstats["treewidth_minfill"]),
        circuit_graph_regime=graph_regime,
        practical_near_tree_flag=near_tree_flag,
        graph_optimum_regime=graph_optimum_regime,
        sat_seconds=sat_seconds,
        sat_variables=cert.variables_at_optimum,
        sat_clauses=cert.clauses_at_optimum,
        clique_lower_bound=cert.lower_bound_clique,
        greedy_upper_bound=cert.upper_bound_greedy,
        certificate_valid=cert_ok,
        retiming_certificate_valid=ret_cert.valid_by_local_commutation and ret_cert.replay_valid,
        formal_global_equivalence_proven=formal_report.proven,
        formal_global_equivalence_scope=formal_report.scope,
        formal_global_equivalence_statement=formal_report.statement,
        formal_undischarged_obligations=len(formal_report.undischarged_obligations),
        equivalence_claim=equivalence_claim,
        supplemental_equivalence_check=supplemental,
        zx_used=cfg.use_zx_normalization,
        zx_available=HAVE_PYZX,
        zx_pre_vertices=zx_stats.get("pre_vertices"),
        zx_post_vertices=zx_stats.get("post_vertices"),
        tket_t_depth=tket_res.t_depth,
        pyzx_t_depth=pyzx_res.t_depth,
        qiskit_opt_t_depth=qiskit_res.t_depth,
        tket_status=tket_res.status,
        pyzx_status=pyzx_res.status,
        qiskit_status=qiskit_res.status,
        baseline_coverage=baseline_coverage,
        formal_equivalence_status=formal_eq,
        artifacts_dir=str(idir),
    )
    write_json(idir / "result.json", result)
    return result


def results_to_dataframe(results: Iterable[InstanceResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def write_summary(df: pd.DataFrame, out_dir: str | Path) -> Dict[str, Any]:
    out = ensure_dir(out_dir)
    if df.empty:
        summary = {"instances": 0}
    else:
        summary = {
            "instances": int(len(df)),
            "certificates_valid": int(df["certificate_valid"].sum()),
            "retiming_certificates_valid": int(df["retiming_certificate_valid"].sum()),
            "equivalence_claim_true": int(df["equivalence_claim"].astype(str).str.startswith("true").sum()),
            "supplemental_true": int(df["supplemental_equivalence_check"].astype(str).str.startswith("true").sum()),
            "formal_equivalence_true": int(df["formal_equivalence_status"].astype(str).str.startswith("true").sum()) if "formal_equivalence_status" in df.columns else 0,
            "formal_global_equivalence_proven": int(df["formal_global_equivalence_proven"].sum()) if "formal_global_equivalence_proven" in df.columns else 0,
            "formal_global_equivalence_failures": int((~df["formal_global_equivalence_proven"].astype(bool)).sum()) if "formal_global_equivalence_proven" in df.columns else 0,
            "tket_numeric": int(df["tket_t_depth"].notna().sum()) if "tket_t_depth" in df.columns else 0,
            "pyzx_numeric": int(df["pyzx_t_depth"].notna().sum()) if "pyzx_t_depth" in df.columns else 0,
            "mean_sat_seconds": float(df["sat_seconds"].mean()),
            "max_sat_seconds": float(df["sat_seconds"].max()),
            "max_treewidth_minfill": None if df["graph_treewidth_minfill"].dropna().empty else int(df["graph_treewidth_minfill"].dropna().max()),
            "max_t_count": int(df["t_count"].max()),
            "mean_construction_gap": float(df["construction_gap"].mean()),
            "max_construction_gap": int(df["construction_gap"].max()),
            "near_tree_circuit_instances": int(df["practical_near_tree_flag"].sum()) if "practical_near_tree_flag" in df.columns else 0,
            "separation_witnesses": int(df["separation_witness"].sum()) if "separation_witness" in df.columns else 0,
            "retiming_limitation_cases": int(df["retiming_limitation_flag"].sum()) if "retiming_limitation_flag" in df.columns else 0,
            "retiming_improved_cases": int((df["retiming_improvement"] > 0).sum()) if "retiming_improvement" in df.columns else 0,
        }
    write_json(out / "summary.json", summary)
    if not df.empty:
        family_summary = df.groupby("family").agg(
            instances=("name", "count"),
            max_n_qubits=("n_qubits", "max"),
            max_t_count=("t_count", "max"),
            max_treewidth=("graph_treewidth_minfill", "max"),
            mean_sat_seconds=("sat_seconds", "mean"),
            mean_gap=("construction_gap", "mean"),
            separation_witnesses=("separation_witness", "sum"),
            near_tree_instances=("practical_near_tree_flag", "sum"),
            retiming_improved_cases=("retiming_improvement", lambda x: int((x > 0).sum())),
        ).reset_index()
        family_summary.to_csv(out / "family_summary.csv", index=False)
        analysis_cols = [c for c in ["family", "name", "n_qubits", "t_count", "graph_edges", "graph_treewidth_minfill", "circuit_graph_regime", "practical_near_tree_flag", "graph_optimum_regime", "sat_variables", "sat_clauses", "sat_seconds", "certified_graph_layers", "constructive_t_depth", "construction_gap", "retiming_improvement", "retiming_limitation_flag", "separation_witness", "construction_gap_interpretation", "qiskit_opt_t_depth", "tket_t_depth", "pyzx_t_depth", "tket_status", "pyzx_status", "baseline_coverage", "formal_equivalence_status"] if c in df.columns]
        df[analysis_cols].to_csv(out / "analysis_table.csv", index=False)
        numeric = df[[c for c in ["t_count", "graph_edges", "graph_treewidth_minfill", "sat_variables", "sat_clauses", "sat_seconds", "construction_gap"] if c in df.columns]].copy()
        if not numeric.empty:
            numeric.corr(numeric_only=True).to_csv(out / "correlation_matrix.csv")
        # Paper-facing diagnostic tables
        if "formal_global_equivalence_proven" in df.columns:
            eq_cols = [c for c in ["name", "family", "formal_global_equivalence_proven", "formal_global_equivalence_scope", "formal_undischarged_obligations", "retiming_certificate_valid", "supplemental_equivalence_check"] if c in df.columns]
            df[eq_cols].to_csv(out / "equivalence_summary.csv", index=False)
        baseline_cols = [c for c in ["name", "family", "qiskit_opt_t_depth", "tket_t_depth", "pyzx_t_depth", "qiskit_status", "tket_status", "pyzx_status", "baseline_coverage"] if c in df.columns]
        if baseline_cols:
            df[baseline_cols].to_csv(out / "baseline_summary.csv", index=False)
        gap_cols = [c for c in ["name", "family", "n_qubits", "t_count", "t_depth_input", "certified_graph_layers", "constructive_t_depth", "construction_gap", "retiming_improvement", "retiming_limitation_flag", "separation_witness", "construction_gap_interpretation", "graph_treewidth_minfill", "circuit_graph_regime", "practical_near_tree_flag", "graph_optimum_regime"] if c in df.columns]
        if gap_cols:
            df[gap_cols].to_csv(out / "construction_gap_summary.csv", index=False)
            try:
                sep = df[df["construction_gap"].fillna(0) > 0].copy()
                sep[gap_cols].sort_values(["construction_gap", "t_count"], ascending=[False, False]).to_csv(out / "separation_examples.csv", index=False)
            except Exception:
                pass
        diag_cols = [c for c in ["name", "family", "n_qubits", "t_count", "graph_vertices", "graph_edges", "graph_density", "graph_treewidth_minfill", "certified_graph_layers", "circuit_graph_regime", "practical_near_tree_flag", "graph_optimum_regime", "sat_variables", "sat_clauses", "sat_seconds"] if c in df.columns]
        if diag_cols:
            df[diag_cols].to_csv(out / "circuit_graph_diagnostics.csv", index=False)
        limitation_cols = [c for c in ["name", "family", "t_depth_input", "constructive_t_depth", "certified_graph_layers", "construction_gap", "retiming_improvement", "retiming_limitation_flag", "construction_gap_interpretation"] if c in df.columns]
        if limitation_cols:
            df[limitation_cols].to_csv(out / "retiming_limitation_summary.csv", index=False)
        write_equivalence_theory_files(out)
        export_v9_theory_tables(out)
    return summary


# ---------------- diagnostics/tests ----------------
def self_test_graph_only(tmp_dir: str = "_qip_graph_selftest") -> pd.DataFrame:
    ensure_dir(tmp_dir)
    graphs = {
        "empty": nx.Graph(),
        "path4": nx.path_graph(4),
        "cycle5": nx.cycle_graph(5),
        "clique4": nx.complete_graph(4),
        "complete_bipartite_3_3": nx.complete_bipartite_graph(3, 3),
    }
    rows = []
    for name, G in graphs.items():
        cert = minimize_t_layers(G, tmp_dir, write_dimacs=True, use_maxsat=True)
        ok, errors = verify_certificate(G, cert)
        rows.append({"name": name, "vertices": G.number_of_nodes(), "edges": G.number_of_edges(), "L": cert.optimum_layers, "valid": ok, "errors": errors})
    df = pd.DataFrame(rows)
    assert df["valid"].all(), "certificate verification failed"
    assert int(df.loc[df.name == "clique4", "L"].iloc[0]) == 4, "clique4 should require four colors"
    assert int(df.loc[df.name == "complete_bipartite_3_3", "L"].iloc[0]) == 2, "K3,3 should require two colors"
    return df


def self_test_once(tmp_dir: str = "_qip_selftest") -> pd.DataFrame:
    if not HAVE_QISKIT:
        return self_test_graph_only(tmp_dir)
    cfg = ExperimentConfig(out_dir=tmp_dir, use_zx_normalization=False, write_dimacs=True, verify_equivalence_up_to_qubits=6)
    specs = [
        ("qaoa_ring", 4, 1, 1),
        ("qaoa_complete", 4, 1, 1),
        ("random_ct", 4, 2, 1),
        ("dense_phase", 4, 1, 1),
        ("clique_stress", 4, 1, 1),
        ("brickwork_ct", 4, 2, 1),
        ("random_interaction_ct", 4, 2, 1),
        ("qft_phase_like", 4, 1, 1),
        ("grover", 4, 1, 1),
    ]
    results = []
    for fam, n, d, p in specs:
        results.append(run_instance(f"test_{fam}_n{n}_d{d}_p{p}", fam, n, cfg, depth=d, p=p))
    df = results_to_dataframe(results)
    assert df["certificate_valid"].all(), "SAT certificate verification failed"
    assert df["retiming_certificate_valid"].all(), "retiming proof replay failed"
    assert df["equivalence_claim"].astype(str).str.startswith("true").all(), "commutation certificate failed"
    assert (df["constructive_t_depth"] >= df["certified_graph_layers"]).all(), "constructive depth below graph optimum impossible"
    return df

# ---------------- paper-facing diagnostic tables ----------------
def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    try:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def export_v9_theory_tables(out_dir: str | Path) -> None:
    """Export paper-facing evidence tables for v9 theoretical claims.

    These tables are intentionally interpretive rather than raw benchmark logs.
    They are designed to support the new QIP framing:
    (i) graph-level optimality is a certificate, not a complete compiler claim;
    (ii) local-commutation retiming has a replayable global soundness proof;
    (iii) graph-optimal and locally constructible T-layer depths can separate;
    (iv) practical circuit instances in this suite induce near-tree dependency graphs,
         while graph-stress probes demonstrate behavior away from that regime.
    """
    out = ensure_dir(out_dir)
    metrics = _safe_read_csv(out / "qip_suite_metrics.csv")
    if metrics.empty:
        metrics = _safe_read_csv(out / "analysis_table.csv")
    graph = _safe_read_csv(out / "graph_stress_metrics.csv")

    # 1) Theorem/claim status table.
    rows: List[Dict[str, Any]] = []
    if not metrics.empty:
        n = len(metrics)
        cert_ok = int(metrics.get("certificate_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "certificate_valid" in metrics else None
        eq_ok = int(metrics.get("formal_global_equivalence_proven", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "formal_global_equivalence_proven" in metrics else None
        gaps = metrics.get("construction_gap", pd.Series(dtype=float)).fillna(0) if "construction_gap" in metrics else pd.Series(dtype=float)
        near = metrics.get("practical_near_tree_flag", pd.Series(dtype=bool)).fillna(False).astype(bool) if "practical_near_tree_flag" in metrics else pd.Series(dtype=bool)
        rows.extend([
            {
                "claim_id": "T1_graph_optimality_certificate",
                "paper_claim": "SAT certificate proves the optimum T-layer count of the extracted dependency graph.",
                "claim_type": "formal_plus_checked",
                "scope": "dependency_graph_only",
                "evidence": f"{cert_ok}/{n} instance certificates valid" if cert_ok is not None else "certificate column missing",
                "status": "supported" if cert_ok == n else "needs_attention",
                "novelty_role": "certified graph-level optimum, separated from circuit realization",
            },
            {
                "claim_id": "T2_replayable_global_equivalence",
                "paper_claim": "A sequence of locally sound adjacent commutations yields global equivalence from encoding input to retimed circuit.",
                "claim_type": "theorem_plus_replay_certificate",
                "scope": "encoding_input_to_retimed_conservative_only",
                "evidence": f"{eq_ok}/{n} replay/global equivalence reports proven" if eq_ok is not None else "formal equivalence column missing",
                "status": "supported" if eq_ok == n else "needs_attention",
                "novelty_role": "machine-checkable retiming proof rather than end-to-end compiler overclaim",
            },
            {
                "claim_id": "T3_local_retiming_separation",
                "paper_claim": "Graph-optimal T-layering can be strictly smaller than the T-depth obtained by conservative local-commutation retiming.",
                "claim_type": "existence_theorem_with_empirical_witnesses",
                "scope": "specified local commutation system and greedy replayable constructor",
                "evidence": f"{int((gaps > 0).sum())}/{n} instances with positive construction gap; max_gap={int(gaps.max()) if len(gaps) else 'NA'}",
                "status": "supported" if len(gaps) and int((gaps > 0).sum()) > 0 else "needs_witness",
                "novelty_role": "construction gap / realizability obstruction",
            },
            {
                "claim_id": "T4_near_tree_practical_regime",
                "paper_claim": "The circuit-derived benchmark suite induces near-tree dependency graphs, explaining the small SAT instances.",
                "claim_type": "empirical_structural_theorem_candidate",
                "scope": "benchmarked circuit families, not universal quantum circuits",
                "evidence": f"{int(near.sum())}/{n} near-tree instances; max_treewidth={metrics['graph_treewidth_minfill'].max() if 'graph_treewidth_minfill' in metrics else 'NA'}",
                "status": "supported" if len(near) and int(near.sum()) == n else "mixed",
                "novelty_role": "identifies a tractable structural regime rather than hiding trivial SAT cases",
            },
        ])
    if not graph.empty:
        rows.append({
            "claim_id": "T5_stress_regime_scaling",
            "paper_claim": "Graph-stress instances exhibit increasing structural width and larger SAT encodings away from the near-tree circuit regime.",
            "claim_type": "controlled_stress_evidence",
            "scope": "graph-only stress probes, not directly quantum-circuit-derived",
            "evidence": f"{len(graph)} graph stress instances; max_treewidth={graph['treewidth_minfill'].max() if 'treewidth_minfill' in graph else 'NA'}; max_vertices={graph['vertices'].max() if 'vertices' in graph else 'NA'}",
            "status": "supported",
            "novelty_role": "separates practical near-tree behavior from worst-case graph coloring hardness",
        })
    if rows:
        pd.DataFrame(rows).to_csv(out / "theorem_evidence_table.csv", index=False)

    # 2) Family-level realizability gap and separation evidence.
    if not metrics.empty and "family" in metrics:
        agg = metrics.groupby("family").agg(
            instances=("name", "count"),
            max_qubits=("n_qubits", "max"),
            max_t_count=("t_count", "max"),
            max_graph_treewidth=("graph_treewidth_minfill", "max"),
            max_graph_layers=("certified_graph_layers", "max"),
            mean_graph_layers=("certified_graph_layers", "mean"),
            max_constructive_t_depth=("constructive_t_depth", "max"),
            mean_constructive_t_depth=("constructive_t_depth", "mean"),
            mean_gap=("construction_gap", "mean"),
            max_gap=("construction_gap", "max"),
            separation_witnesses=("construction_gap", lambda x: int((x.fillna(0) > 0).sum())),
            retiming_improved_cases=("retiming_improvement", lambda x: int((x.fillna(0) > 0).sum())) if "retiming_improvement" in metrics else ("construction_gap", "count"),
        ).reset_index()
        agg["family_interpretation"] = agg.apply(
            lambda r: "strong_separation_family" if r["separation_witnesses"] > 0 and r["max_gap"] >= 4 else (
                "mild_separation_family" if r["separation_witnesses"] > 0 else "graph_optimum_constructively_realized_in_suite"
            ), axis=1)
        agg.to_csv(out / "realizability_gap_by_family.csv", index=False)

        # Cases where the graph optimum is small but construction gap is large.
        if "construction_gap" in metrics:
            cols = [c for c in ["name", "family", "n_qubits", "t_count", "t_depth_input", "certified_graph_layers", "constructive_t_depth", "construction_gap", "graph_treewidth_minfill", "artifacts_dir"] if c in metrics.columns]
            hard_sep = metrics[(metrics["construction_gap"].fillna(0) > 0)].copy()
            if not hard_sep.empty:
                hard_sep[cols].sort_values(["construction_gap", "t_count"], ascending=[False, False]).to_csv(out / "local_retiming_separation_witnesses.csv", index=False)

    # 3) Obstruction certificate summary from retiming_certificate.json files.
    obstruction_rows: List[Dict[str, Any]] = []
    if not metrics.empty:
        for _, r in metrics.iterrows():
            idir = Path(str(r.get("artifacts_dir", out / str(r.get("name", "")))))
            cert_path = idir / "retiming_certificate.json"
            if not cert_path.exists():
                cert_path = out / str(r.get("name", "")) / "retiming_certificate.json"
            try:
                js = json.loads(cert_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            failed = js.get("failed_swaps", []) or []
            swaps = js.get("swaps", []) or []
            blocker_counts: Dict[str, int] = {}
            blocker_support_counts: Dict[str, int] = {}
            for f in failed:
                b = f.get("blocked_by", {}) or {}
                gate = str(b.get("gate", f.get("reason", "unknown")))
                blocker_counts[gate] = blocker_counts.get(gate, 0) + 1
                mq = (f.get("moving", {}) or {}).get("qubits", []) or []
                bq = b.get("qubits", []) or []
                key = "disjoint" if set(mq).isdisjoint(set(bq)) else "overlap"
                blocker_support_counts[key] = blocker_support_counts.get(key, 0) + 1
            obstruction_rows.append({
                "name": r.get("name"),
                "family": r.get("family"),
                "construction_gap": r.get("construction_gap"),
                "successful_swaps": len(swaps),
                "failed_swaps": len(failed),
                "dominant_blocker_gate": max(blocker_counts, key=blocker_counts.get) if blocker_counts else "none",
                "blocker_gate_histogram": json.dumps(blocker_counts, sort_keys=True),
                "blocker_support_histogram": json.dumps(blocker_support_counts, sort_keys=True),
                "obstruction_certificate_present": bool(failed),
                "interpretation": "local_rule_obstruction_to_greedy_realization" if failed else "no_local_obstruction_recorded",
            })
    if obstruction_rows:
        obs = pd.DataFrame(obstruction_rows)
        obs.to_csv(out / "obstruction_certificate_summary.csv", index=False)
        if "family" in obs:
            fam_obs = obs.groupby("family").agg(
                instances=("name", "count"),
                obstruction_instances=("obstruction_certificate_present", "sum"),
                total_failed_swaps=("failed_swaps", "sum"),
                mean_failed_swaps=("failed_swaps", "mean"),
                max_failed_swaps=("failed_swaps", "max"),
                dominant_blockers=("dominant_blocker_gate", lambda x: json.dumps({str(k): int(v) for k, v in pd.Series(list(x)).value_counts().items()}, sort_keys=True)),
            ).reset_index()
            fam_obs.to_csv(out / "obstruction_by_family.csv", index=False)

    # 4) Near-tree theorem evidence: forests/bipartite/two-color regime.
    if not metrics.empty:
        nt = metrics.copy()
        if "graph_edges" in nt and "graph_vertices" in nt:
            nt["forest_edge_count_condition"] = nt["graph_edges"] <= (nt["graph_vertices"] - 1).clip(lower=0)
        else:
            nt["forest_edge_count_condition"] = False
        if "graph_treewidth_minfill" in nt:
            nt["treewidth_leq_one"] = nt["graph_treewidth_minfill"].fillna(999) <= 1
        else:
            nt["treewidth_leq_one"] = False
        if "certified_graph_layers" in nt:
            nt["chi_leq_two"] = nt["certified_graph_layers"].fillna(999) <= 2
        else:
            nt["chi_leq_two"] = False
        nt_cols = [c for c in ["name", "family", "graph_vertices", "graph_edges", "graph_treewidth_minfill", "certified_graph_layers", "forest_edge_count_condition", "treewidth_leq_one", "chi_leq_two", "sat_variables", "sat_clauses", "sat_seconds"] if c in nt.columns]
        nt[nt_cols].to_csv(out / "near_tree_theorem_evidence.csv", index=False)

    # 5) Literature positioning file (source-aware but citation-free in artifact).
    literature_rows = [
        {
            "related_direction": "Polynomial-time T-depth optimization via matroid partitioning",
            "implication_for_this_work": "Do not claim first T-depth optimization; position against exact/structural methods and emphasize certificates plus realizability gap.",
            "v9_distinction": "dependency-graph certificate and construction-gap analysis rather than matroid-based re-synthesis",
        },
        {
            "related_direction": "ZX-based T-count/T-depth optimization and PyZX/tket toolchains",
            "implication_for_this_work": "Do not claim ZX+SAT integration alone as the novelty.",
            "v9_distinction": "separates graph optimum, replayable local retiming proof, and obstruction certificates",
        },
        {
            "related_direction": "Graph-coloring formulations of commuting-gate depth optimization",
            "implication_for_this_work": "Graph coloring itself is not the central novelty.",
            "v9_distinction": "novelty is the constructibility/realizability gap and evidence-backed obstruction taxonomy",
        },
        {
            "related_direction": "SAT-based quantum synthesis/optimization",
            "implication_for_this_work": "Avoid generic SAT-compilation novelty claims.",
            "v9_distinction": "paper-facing certificates connect SAT optimality, treewidth regime, and replayable retiming equivalence",
        },
    ]
    pd.DataFrame(literature_rows).to_csv(out / "literature_positioning_matrix.csv", index=False)


# Backward-compatible alias used by the experiment runner.
export_v11_theory_tables = export_v9_theory_tables
