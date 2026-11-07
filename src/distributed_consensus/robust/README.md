# `robust/` — scalar W-MSR filtering and graph-robustness verification

This subpackage implements the simultaneous local scalar **Weighted-Mean-Subsequence-Reduced (W-MSR)** filtering rule, adversary-set validation, and exact small-graph $(r,s)$-robustness checks used by the adversarial consensus test bench. It does not modify the physical plant, controller parameterization, or small-gain training code in `control/` or `learning/`.

## `wmsr.py` — local scalar W-MSR update

`wmsr_local(X, N, F=1)` and `wmsr_local_with_details(X, N, F=1)` operate on a 1D finite scalar signal vector $X \in \mathbb{R}^N$ ($N \ge 2$), a zero-based neighbor mapping $\mathcal{N}: \{0,\dots,N-1\} \to 2^{\{0,\dots,N-1\}}$, and a nonnegative integer trimming parameter $F \in \mathbb{Z}_{\ge 0}$.

### Simultaneous trimming and averaging rule

At each discrete step, every node $i \in \{0,\dots,N-1\}$ orders its neighbor indices $j \in \mathcal{N}_i$ by $(X_j, j)$ and partitions them relative to its own scalar value $X_i$:

$$
\mathcal{L}_i = \{j \in \mathcal{N}_i : X_j < X_i\}, \qquad
\mathcal{M}_i = \{j \in \mathcal{N}_i : X_j = X_i\}, \qquad
\mathcal{U}_i = \{j \in \mathcal{N}_i : X_j > X_i\}.
$$

1. **Lower trim (`removed_low_ids`):** Discard the $\min(F, |\mathcal{L}_i|)$ smallest values strictly below $X_i$, denoted $\mathcal{R}_i^- = \mathcal{L}_i[:F]$, leaving $\mathcal{L}_i^{\mathrm{ret}} = \mathcal{L}_i[F:]$.
2. **Upper trim (`removed_high_ids`):** Discard the $\min(F, |\mathcal{U}_i|)$ largest values strictly above $X_i$, denoted $\mathcal{R}_i^+ = \mathcal{U}_i[-F:]$, leaving $\mathcal{U}_i^{\mathrm{ret}} = \mathcal{U}_i[:-F]$ (or $\mathcal{U}_i$ when $F=0$).
3. **Equal values:** All neighbors $j \in \mathcal{M}_i$ with $X_j = X_i$ are retained.
4. **Retained neighbor set (`retained_ids`):**

$$
\mathcal{K}_i = \mathcal{L}_i^{\mathrm{ret}} \cup \mathcal{M}_i \cup \mathcal{U}_i^{\mathrm{ret}}.
$$

5. **Uniform average update (`values`):**

$$
X_i^+ = \begin{cases}
\dfrac{1}{1 + |\mathcal{K}_i|}\left(X_i + \displaystyle\sum_{j \in \mathcal{K}_i} X_j\right), & |\mathcal{K}_i| > 0, \\[10pt]
X_i, & |\mathcal{K}_i| = 0.
\end{cases}
$$

All updates are synchronous: $X^+$ is computed from the pre-update vector $X$ without in-place mutation. `wmsr_local_with_details()` returns a frozen `WMSRLocalResult` dataclass containing `values`, `retained_ids`, `removed_high_ids`, `removed_low_ids`, and the union property `removed_ids`.

### Integration with scalar-projected feedback controllers

In `simulation/model.py`, each agent $i$ broadcasts a single scalar coordination signal $s_i = p^T x_i \in \mathbb{R}$ ($p = W$ for the small-gain controller; $p = H^T$ for the 1-channel paper GNN). From the W-MSR output $s_i^+ = X_i^+$ and retained count $r_i = |\mathcal{K}_i| > 0$, the retained-neighbor average and effective retained disagreement are recovered algebraically as:

$$
\bar{s}_{\mathcal{K}_i} = \frac{(r_i + 1)s_i^+ - s_i}{r_i}, \qquad
e_{i,\mathrm{rw}}^{\mathrm{ret}} = s_i - \bar{s}_{\mathcal{K}_i}, \qquad
e_{i,\mathrm{un}}^{\mathrm{ret}} = \sum_{j \in \mathcal{K}_i}(s_i - s_j) = (r_i + 1)(s_i - s_i^+).
$$

## Threat models and adversary validation

`validate_adversary_set(graph, adversary_ids, F, threat_model)` validates the compromised node set $\mathcal{A} \subset \mathcal{V}$ ($|\mathcal{A}| < N$) against the declared fault model:

- **`F-total-malicious` / `F-total-byzantine`:** At most $F$ compromised agents exist globally:

$$
|\mathcal{A}| \le F.
$$

- **`F-local-malicious` / `F-local-byzantine`:** Every normal agent $i \in \mathcal{V} \setminus \mathcal{A}$ has at most $F$ compromised neighbors:

$$
|\mathcal{N}_i \cap \mathcal{A}| \le F \quad \forall\, i \in \mathcal{V} \setminus \mathcal{A}.
$$

## Exact graph-robustness verification

For a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, a nonempty subset $S \subset \mathcal{V}$, and an integer $r \ge 1$, let $\mathcal{X}_S^r$ denote the set of nodes in $S$ that have at least $r$ neighbors outside $S$:

$$
\mathcal{X}_S^r = \left\{i \in S : |\mathcal{N}_i \setminus S| \ge r\right\}.
$$

- **`is_r_robust(graph, r, max_nodes=14)`:** Tests whether $\mathcal{G}$ is **$r$-robust**, meaning that for every pair of nonempty disjoint subsets $S_1, S_2 \subset \mathcal{V}$ ($S_1 \cap S_2 = \emptyset$), at least one satisfies:

$$
|\mathcal{X}_{S_1}^r| \ge 1 \quad \text{or} \quad |\mathcal{X}_{S_2}^r| \ge 1.
$$

- **`is_rs_robust(graph, r, s, max_nodes=14)`:** Tests whether $\mathcal{G}$ is **$(r,s)$-robust** ($1 \le s \le N$), meaning that for every pair of nonempty disjoint subsets $S_1, S_2 \subset \mathcal{V}$, at least one of the following holds:

$$
|\mathcal{X}_{S_1}^r| = |S_1|, \qquad |\mathcal{X}_{S_2}^r| = |S_2|, \qquad \text{or} \qquad |\mathcal{X}_{S_1}^r| + |\mathcal{X}_{S_2}^r| \ge s.
$$

- **`assess_wmsr_topology(graph, F, threat_model, adversary_ids=(), max_nodes=14)`:** Evaluates the textbook fixed-graph condition corresponding to `threat_model`:
  - `F-total-malicious`: checks $(F+1, F+1)$-robustness on $\mathcal{G}$ (necessary and sufficient for scalar W-MSR).
  - `F-local-malicious`: checks $(2F+1)$-robustness on $\mathcal{G}$ (sufficient condition; $(F+1)$-robustness is necessary).
  - `F-total-byzantine` / `F-local-byzantine`: checks $(F+1)$-robustness on the normal-node induced subgraph $\mathcal{G}[\mathcal{V} \setminus \mathcal{A}]$.

### Scope and limitations

- **Exponential bitmask complexity:** Enumerating all disjoint subset pairs $(S_1, S_2)$ requires $\mathcal{O}(3^N)$ operations. Above `max_nodes=14`, `is_r_robust()` and `is_rs_robust()` return `None` (`exact_check=False`) rather than mistaking connectivity or minimum degree for robustness.
- **No automatic closed-loop plant guarantee:** The $(r,s)$-robustness conditions are textbook guarantees for first-order scalar consensus $x_i(k+1) = X_i^+(k)$. Applying scalar W-MSR inside a two-state heterogeneous feedback loop makes the effective graph operator state-dependent, so neither the scalar W-MSR theorem nor the fixed-Laplacian small-gain bound $\gamma_{\mathrm{plant}}\|W\|_2 < 1$ automatically certifies closed-loop stability under attack.