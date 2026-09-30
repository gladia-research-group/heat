# Machine-checked proofs

`HEEquivQAT.lean` formalizes the appendix *(F)HE-Aware Training and
Quantization-Aware Training*: learning approximation depth is quantization-aware
training with the bit width set by the iteration count.

Three claims, in the order the argument needs them:

- **Iterations are bits.** `quantizer_error_eq_iterator_error` matches the
  worst-case error of a uniform `b`-bit quantizer to that of an `n`-step
  iterator, forcing `b = log₂(1 + R / 2ε_n)`;
  `quantizer_error_eq_iterator_error_iff` proves this `b` is the *only* one —
  the "if and only if" of the restated proposition. `bitwidth_linear` and
  `matchedBits_isTheta` give `b = Θ(2ⁿ)`, and `matchedBits_ratio_tendsto_two`
  states it as the paper does — each iteration roughly doubles the correct bits.
  `matchedBits_remainder_bound` quantifies the step the paper takes with `≈`.
- **Perturbation reaches activations only through the linear map.**
  `act_perturb`, then `trajectory_bound` (discrete Grönwall) and `traj_dist_le`
  for how it accumulates across depth.
- **The objective bound.** `heat_is_qat`: the HEAT and QAT objectives differ by
  at most `2 ξ M r^(2ⁿ) Σⱼ ρʲ`, and `heat_is_qat_bound_tendsto_zero` shows that
  vanishes doubly exponentially in `n`.

`heat_is_qat` keeps layers abstract. `heat_is_qat_layers` closes the loop on the
paper's concrete layer `h ↦ (g(Σⱼ Wᵢⱼ hⱼ))ᵢ`, *proving* rather than assuming the
per-layer budgets — Lipschitz constant `ρ = λκ` from `layer_lipschitz`,
approximation error from `layer_approx_error`, rounding error from
`layer_quant_error`, and the two matched by `matched_step_size`. Its hypotheses
are exactly the appendix's: the iterator bound `|Fₙ x − f x| ≤ εₙ` is assumed
only on the calibrated domain `D`, every pre-activation of the HEAT trajectory
is assumed to lie in `D`, and the quantizer sits at the paper's integer matched
bit width `b*(n)`, i.e. its per-layer error is *at most* `εₙ`
(`hbudget : λ (s/2) d B ≤ εₙ`).

- **Per-site counts.** The appendix's two extensions (its equations *fidelity*
  and *mixed-precision*) are `fidelity_bound_sites` — the deployed circuit with
  counts `n k` is within `ξ Σₖ ρ^(L−1−k) M r^(2^(n k))` of the exact network — and
  `heat_is_qat_sites`, the same against QAT at the per-site matched bit widths,
  with the factor 2. Both rest on `trajectory_bound_sites` (the Grönwall
  induction with a site-dependent budget) and `traj_dist_le_sites`;
  `fidelity_bound_layers_sites` and `heat_is_qat_layers_sites` are the
  concrete-layer versions. Sites are 0-indexed (site `k` maps `h_k ↦ h_{k+1}`), so
  the paper's `Σ_{ℓ=1}^{L} ρ^{L−ℓ} ε_{n_ℓ}` reads `Σ_{k<L} ρ^{L−1−k} ε k` here;
  `sum_sites_const` recovers the uniform-count form `εₙ Σ_{j<L} ρʲ`.

Not formalized: the remark that the smallest integer `b` with
`λ s(b)/2 · d · B ≤ εₙ` equals `⌈bₙ(λ d B R)⌉` (a monotonicity statement about
`s(b)`; `matched_step_size` gives the real-valued solution), and the *rate*
"doubly exponentially" beyond the limit in `heat_is_qat_bound_tendsto_zero`.

## Verifying

```bash
cd proofs
lake exe cache get     # prebuilt mathlib oleans
lake build
```

Pinned to Lean v4.28.0 and mathlib `8f9d9cf` (resolved `8f9d9cff6bd7`). Verified
against that pair: `lake build` completes with no errors, the source contains no
`sorry` and declares no axioms of its own, and every one of the 25 public
theorems depends only on Lean's three standard axioms — `propext`,
`Classical.choice`, `Quot.sound`. Other mathlib revisions may need the proof
scripts adjusted.
