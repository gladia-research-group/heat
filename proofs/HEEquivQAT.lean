import Mathlib

namespace HEEquivQAT

open Finset

/- Errors definition -/
noncomputable def eps (M r : ℝ) (n : ℕ) : ℝ := M * r ^ (2 ^ n) /- eps_n -/
noncomputable def step (R b : ℝ) : ℝ := R / ((2 : ℝ) ^ b - 1) /- R /(2^b -1) -/
noncomputable def matchedBits (R M r : ℝ) (n : ℕ) : ℝ := Real.logb 2 (1 + R / (2 * eps M r n)) /- log_2(1+ R / (2 eps_n) ) -/

/- Proposition: iterations are bits -/
theorem quantizer_error_eq_iterator_error
    (R M r : ℝ) (n : ℕ) (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) :
    step R (matchedBits R M r n) / 2 = eps M r n := by
  have he : 0 < eps M r n := mul_pos hM (pow_pos hr0 _)
  rw [step, matchedBits, Real.rpow_logb (by norm_num) (by norm_num) (by positivity),
    add_sub_cancel_left, div_div_eq_mul_div]
  field_simp

theorem quantizer_error_eq_iterator_error_iff
    (R M r b : ℝ) (n : ℕ) (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) :
    step R b / 2 = eps M r n ↔ b = matchedBits R M r n := by
  have he : 0 < eps M r n := mul_pos hM (pow_pos hr0 _)
  constructor
  · intro hmatch
    have hstep2 : R / ((2 : ℝ) ^ b - 1) = 2 * eps M r n := by
      rw [step] at hmatch
      linarith
    have hden : (2 : ℝ) ^ b - 1 ≠ 0 := by
      intro h
      rw [h, div_zero] at hstep2
      linarith
    have hpow : (2 : ℝ) ^ b = 1 + R / (2 * eps M r n) := by
      rw [div_eq_iff hden] at hstep2
      have hne : (2 * eps M r n) ≠ 0 := by positivity
      field_simp
      nlinarith [hstep2]
    have hlog : Real.logb 2 ((2 : ℝ) ^ b) = b := Real.logb_rpow (by norm_num) (by norm_num)
    rw [← hlog, hpow, matchedBits]
  · intro hb
    rw [hb]
    exact quantizer_error_eq_iterator_error R M r n hR hM hr0

theorem bitwidth_linear (R M r : ℝ) (n : ℕ) (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) :
    Real.logb 2 (R / (2 * eps M r n))
      = (2 : ℝ) ^ n * Real.logb 2 (1 / r) + Real.logb 2 (R / (2 * M)) := by
  have h1 : R / (2 * eps M r n) = (R / (2 * M)) * (1 / r) ^ (2 ^ n) := by
    rw [eps, div_pow, one_pow]
    field_simp
  rw [h1, Real.logb_mul (by positivity) (by positivity), Real.logb_pow]
  push_cast
  ring

theorem matchedBits_eq_affine_add_remainder (R M r : ℝ) (n : ℕ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) :
    matchedBits R M r n
      = (2 : ℝ) ^ n * Real.logb 2 (1 / r) + Real.logb 2 (R / (2 * M))
        + Real.logb 2 (1 + 2 * eps M r n / R) := by
  have he : 0 < eps M r n := mul_pos hM (pow_pos hr0 _)
  have hkey : 1 + R / (2 * eps M r n)
      = (R / (2 * eps M r n)) * (1 + 2 * eps M r n / R) := by
    field_simp
    ring
  rw [matchedBits, hkey, Real.logb_mul (by positivity) (by positivity),
    bitwidth_linear R M r n hR hM hr0]

theorem matchedBits_remainder_bound (R M r : ℝ) (n : ℕ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) :
    0 ≤ Real.logb 2 (1 + 2 * eps M r n / R)
      ∧ Real.logb 2 (1 + 2 * eps M r n / R) ≤ 2 * eps M r n / (R * Real.log 2) := by
  have he : 0 < eps M r n := mul_pos hM (pow_pos hr0 _)
  set x : ℝ := 2 * eps M r n / R with hx
  have hx0 : 0 < x := by positivity
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hle : Real.log (1 + x) ≤ x := by
    have := Real.log_le_sub_one_of_pos (x := 1 + x) (by linarith)
    linarith
  refine ⟨Real.logb_nonneg (by norm_num) (by linarith), ?_⟩
  have hrw : 2 * eps M r n / (R * Real.log 2) = x / Real.log 2 := by
    rw [hx]; field_simp
  rw [hrw, Real.logb]
  exact (div_le_div_iff_of_pos_right hlog2).mpr hle

private lemma eps_tendsto_zero (M r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) :
    Filter.Tendsto (fun n : ℕ => eps M r n) Filter.atTop (nhds 0) := by
  have hb : Filter.Tendsto (fun n : ℕ => r ^ n) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hr0.le hr1
  have h2 : Filter.Tendsto (fun n : ℕ => r ^ (2 ^ n)) Filter.atTop (nhds 0) := by
    refine squeeze_zero (fun n => by positivity) (fun n => ?_) hb
    exact pow_le_pow_of_le_one hr0.le hr1.le (Nat.le_of_lt Nat.lt_two_pow_self)
  simpa [eps] using h2.const_mul M

private lemma matchedBits_div_tendsto (R M r : ℝ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) (hr1 : r < 1) :
    Filter.Tendsto (fun n : ℕ => matchedBits R M r n / (2 : ℝ) ^ n) Filter.atTop
      (nhds (Real.logb 2 (1 / r))) := by
  set A := Real.logb 2 (1 / r) with hA
  set C := Real.logb 2 (R / (2 * M)) with hC
  have hrho : Filter.Tendsto (fun n : ℕ => Real.logb 2 (1 + 2 * eps M r n / R))
      Filter.atTop (nhds 0) := by
    refine squeeze_zero (fun n => (matchedBits_remainder_bound R M r n hR hM hr0).1)
      (fun n => (matchedBits_remainder_bound R M r n hR hM hr0).2) ?_
    simpa using ((eps_tendsto_zero M r hr0 hr1).const_mul 2).div_const (R * Real.log 2)
  have hpow : Filter.Tendsto (fun n : ℕ => (1 : ℝ) / (2 : ℝ) ^ n) Filter.atTop (nhds 0) := by
    have h := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num : (1 : ℝ) / 2 < 1)
    exact h.congr (fun n => by rw [div_pow, one_pow])
  have key : Filter.Tendsto (fun n : ℕ => A + (C * (1 / (2 : ℝ) ^ n)
      + Real.logb 2 (1 + 2 * eps M r n / R) * (1 / (2 : ℝ) ^ n))) Filter.atTop (nhds A) := by
    simpa using tendsto_const_nhds.add
      (((tendsto_const_nhds (x := C)).mul hpow).add (hrho.mul hpow))
  refine key.congr (fun n => ?_)
  rw [matchedBits_eq_affine_add_remainder R M r n hR hM hr0]
  have h2 : ((2 : ℝ) ^ n) ≠ 0 := by positivity
  field_simp
  ring

private lemma logb_one_div_pos (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) :
    0 < Real.logb 2 (1 / r) := by
  apply Real.logb_pos (by norm_num)
  rw [lt_div_iff₀ hr0]; linarith

/-- Consequently the matched bit width is `Θ(2^n)`, as claimed by Prop. iters-are-bits. -/
theorem matchedBits_isTheta (R M r : ℝ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) (hr1 : r < 1) :
    Asymptotics.IsTheta Filter.atTop
      (fun n : ℕ => matchedBits R M r n) (fun n : ℕ => ((2 : ℝ) ^ n)) := by
  have hA := logb_one_div_pos r hr0 hr1
  exact (Asymptotics.isTheta_of_div_tendsto_nhds_ne_zero
    (matchedBits_div_tendsto R M r hR hM hr0 hr1) hA.ne').symm

theorem matchedBits_ratio_tendsto_two (R M r : ℝ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) (hr1 : r < 1) :
    Filter.Tendsto (fun n : ℕ => matchedBits R M r (n + 1) / matchedBits R M r n)
      Filter.atTop (nhds 2) := by
  have hA := logb_one_div_pos r hr0 hr1
  set A := Real.logb 2 (1 / r) with hAdef
  have hu : Filter.Tendsto (fun n : ℕ => matchedBits R M r n / (2 : ℝ) ^ n)
      Filter.atTop (nhds A) := matchedBits_div_tendsto R M r hR hM hr0 hr1
  have hu' : Filter.Tendsto (fun n : ℕ => matchedBits R M r (n + 1) / (2 : ℝ) ^ (n + 1))
      Filter.atTop (nhds A) := hu.comp (Filter.tendsto_add_atTop_nat 1)
  have hdiv : Filter.Tendsto (fun n : ℕ => (2 * (matchedBits R M r (n + 1) / (2 : ℝ) ^ (n + 1)))
      / (matchedBits R M r n / (2 : ℝ) ^ n)) Filter.atTop (nhds ((2 * A) / A)) :=
    (hu'.const_mul 2).div hu hA.ne'
  have hval : (2 * A) / A = 2 := by field_simp
  rw [hval] at hdiv
  refine hdiv.congr (fun n => ?_)
  rcases eq_or_ne (matchedBits R M r n) 0 with h | h
  · simp [h]
  · have h2 : ((2 : ℝ) ^ n) ≠ 0 := by positivity
    field_simp
    ring

/- Lemmas for the objective bound: Step 2 (act-perturb) and Step 3 (trajectory bound) -/

theorem act_perturb {d : ℕ} (f : ℝ → ℝ) (lam s B : ℝ) (w q h : Fin d → ℝ)
    (hlam : 0 ≤ lam) (hs : 0 ≤ s) (hB : 0 ≤ B)
    (hf : ∀ x y : ℝ, |f x - f y| ≤ lam * |x - y|)
    (hw : ∀ i, |w i - q i| ≤ s / 2) (hh : ∀ i, |h i| ≤ B) :
    |f (∑ i, w i * h i) - f (∑ i, q i * h i)| ≤ lam * (s / 2) * d * B := by
  have key : |(∑ i, w i * h i) - ∑ i, q i * h i| ≤ (s / 2) * d * B := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ i, (w i * h i - q i * h i)| ≤ ∑ i, |w i * h i - q i * h i| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin d, (s / 2) * B := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [← sub_mul, abs_mul]
          exact mul_le_mul (hw i) (hh i) (abs_nonneg _) (by linarith)
      _ = (s / 2) * d * B := by
          simp [Finset.sum_const]
          ring
  calc |f (∑ i, w i * h i) - f (∑ i, q i * h i)|
      ≤ lam * |(∑ i, w i * h i) - ∑ i, q i * h i| := hf _ _
    _ ≤ lam * ((s / 2) * d * B) := mul_le_mul_of_nonneg_left key hlam
    _ = lam * (s / 2) * d * B := by ring


/- Step 3 lemma: discrete Gronwall -/
theorem trajectory_bound (δ : ℕ → ℝ) (ρ ε : ℝ) (L : ℕ)
    (hρ : 0 ≤ ρ) (hε : 0 ≤ ε) (h0 : δ 0 = 0)
    (hstep : ∀ ℓ, ℓ < L → δ (ℓ + 1) ≤ ρ * δ ℓ + ε) :
    ∀ ℓ, ℓ ≤ L → δ ℓ ≤ ε * ∑ j ∈ range ℓ, ρ ^ j := by
  intro ℓ
  induction ℓ with
  | zero => intro _; simp [h0]
  | succ k ih =>
      intro hk
      have h1 := hstep k (Nat.lt_of_succ_le hk)
      have h2 := ih (Nat.le_of_succ_le hk)
      have h3 : ρ * δ k ≤ ρ * (ε * ∑ j ∈ range k, ρ ^ j) := mul_le_mul_of_nonneg_left h2 hρ
      rw [geom_sum_succ]
      nlinarith

variable {E : Type*} [NormedAddCommGroup E]

def traj (F : ℕ → E → E) (h₀ : E) : ℕ → E
  | 0 => h₀
  | ℓ + 1 => F ℓ (traj F h₀ ℓ)

/- Step 3: accumulation across depth, budget at every input (HEAT side) -/
theorem traj_dist_le (Lex F : ℕ → E → E) (h₀ : E) (ρ ε : ℝ)
    (hρ : 0 ≤ ρ) (hε : 0 ≤ ε)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hpert : ∀ ℓ, ∀ x : E, ‖F ℓ x - Lex ℓ x‖ ≤ ε) (L : ℕ) :
    ‖traj F h₀ L - traj Lex h₀ L‖ ≤ ε * ∑ j ∈ range L, ρ ^ j := by
  refine trajectory_bound (fun ℓ => ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖) ρ ε L hρ hε (by simp [traj])
    (fun ℓ _ => ?_) L le_rfl
  show ‖traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)‖ ≤ ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε
  have h1 : traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)
      = (F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)) := by
    simp [traj]
  rw [h1]
  calc ‖(F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ))‖
      ≤ ‖F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ)‖
        + ‖Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)‖ := norm_add_le _ _
    _ ≤ ε + ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ := add_le_add (hpert ℓ _) (hlip ℓ _ _)
    _ = ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε := by ring

/- Steps 3–4 in the abstract: budgets assumed. The paper theorem is heat_is_qat_layers -/

theorem heat_is_qat (Lex FH FQ : ℕ → E → E) (h₀ : E) (loss : E → ℝ)
    (ρ M r ξ : ℝ) (n L : ℕ)
    (hρ : 0 ≤ ρ) (hM : 0 ≤ M) (hr : 0 ≤ r) (hξ : 0 ≤ ξ)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hH : ∀ ℓ, ∀ x : E, ‖FH ℓ x - Lex ℓ x‖ ≤ eps M r n)
    (hQ : ∀ ℓ, ∀ x : E, ‖FQ ℓ x - Lex ℓ x‖ ≤ eps M r n)
    (hloss : ∀ x y : E, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj FH h₀ L) - loss (traj FQ h₀ L)|
      ≤ 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, ρ ^ j := by
  have hεnn : 0 ≤ eps M r n := mul_nonneg hM (pow_nonneg hr _)
  have hH' := traj_dist_le Lex FH h₀ ρ (eps M r n) hρ hεnn hlip hH L
  have hQ' := traj_dist_le Lex FQ h₀ ρ (eps M r n) hρ hεnn hlip hQ L
  have htri : ‖traj FH h₀ L - traj FQ h₀ L‖
      ≤ ‖traj FH h₀ L - traj Lex h₀ L‖ + ‖traj FQ h₀ L - traj Lex h₀ L‖ := by
    rw [← norm_neg (traj FQ h₀ L - traj Lex h₀ L),
      show traj FH h₀ L - traj FQ h₀ L
        = (traj FH h₀ L - traj Lex h₀ L) + -(traj FQ h₀ L - traj Lex h₀ L) by abel]
    exact norm_add_le _ _
  have h2 := hloss (traj FH h₀ L) (traj FQ h₀ L)
  have h3 : ξ * ‖traj FH h₀ L - traj FQ h₀ L‖
      ≤ 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, ρ ^ j := by
    calc ξ * ‖traj FH h₀ L - traj FQ h₀ L‖
        ≤ ξ * (eps M r n * (∑ j ∈ range L, ρ ^ j) + eps M r n * ∑ j ∈ range L, ρ ^ j) :=
          mul_le_mul_of_nonneg_left (htri.trans (add_le_add hH' hQ')) hξ
      _ = 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, ρ ^ j := by rw [eps]; ring
  linarith

theorem heat_is_qat_bound_tendsto_zero (ρ M r ξ : ℝ) (L : ℕ)
    (hr0 : 0 ≤ r) (hr1 : r < 1) :
    Filter.Tendsto
      (fun n : ℕ => 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, ρ ^ j)
      Filter.atTop (nhds 0) := by
  have h1 : Filter.Tendsto (fun n : ℕ => r ^ (2 ^ n)) Filter.atTop (nhds 0) :=
    (tendsto_pow_atTop_nhds_zero_of_lt_one hr0 hr1).comp
      (tendsto_pow_atTop_atTop_of_one_lt (by norm_num))
  simpa using ((h1.const_mul M).const_mul (2 * ξ)).mul_const (∑ j ∈ range L, ρ ^ j)

/- The theorem for classic layers -/

def layer {d : ℕ} (g : ℝ → ℝ) (W : Fin d → Fin d → ℝ) (h : Fin d → ℝ) : Fin d → ℝ :=
  fun i => g (∑ j, W i j * h j)

/- Step 0: exact layers are (lam * kappa)-Lipschitz -/
theorem layer_lipschitz {d : ℕ} (g : ℝ → ℝ) (W : Fin d → Fin d → ℝ) (lam kappa : ℝ)
    (hlam : 0 ≤ lam) (hg : ∀ x y : ℝ, |g x - g y| ≤ lam * |x - y|)
    (hW : ∀ i, ∑ j, |W i j| ≤ kappa) (x y : Fin d → ℝ) :
    ‖layer g W x - layer g W y‖ ≤ lam * kappa * ‖x - y‖ := by
  rcases Nat.eq_zero_or_pos d with hd | hd
  · subst hd
    have hz : ∀ z : Fin 0 → ℝ, ‖z‖ = 0 := fun z => by
      simpa using Subsingleton.elim z 0
    simp [hz]
  · have hxy : (0 : ℝ) ≤ ‖x - y‖ := norm_nonneg _
    have hkappa : 0 ≤ kappa := by
      have i : Fin d := ⟨0, hd⟩
      exact le_trans (Finset.sum_nonneg fun j _ => abs_nonneg _) (hW i)
    refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun i => ?_
    have h1 : |g (∑ j, W i j * x j) - g (∑ j, W i j * y j)|
        ≤ lam * |(∑ j, W i j * x j) - ∑ j, W i j * y j| := hg _ _
    have h2 : |(∑ j, W i j * x j) - ∑ j, W i j * y j| ≤ (∑ j, |W i j|) * ‖x - y‖ := by
      rw [← Finset.sum_sub_distrib, Finset.sum_mul]
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ => ?_)
      rw [← mul_sub, abs_mul]
      refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
      have := norm_le_pi_norm (x - y) j
      simpa [Real.norm_eq_abs] using this
    have h3 : lam * |(∑ j, W i j * x j) - ∑ j, W i j * y j|
        ≤ lam * ((∑ j, |W i j|) * ‖x - y‖) := mul_le_mul_of_nonneg_left h2 hlam
    have h4 : lam * ((∑ j, |W i j|) * ‖x - y‖) ≤ lam * (kappa * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (hW i) hxy) hlam
    have : |g (∑ j, W i j * x j) - g (∑ j, W i j * y j)| ≤ lam * kappa * ‖x - y‖ := by
      calc |g (∑ j, W i j * x j) - g (∑ j, W i j * y j)| ≤ _ := h1
        _ ≤ _ := h3
        _ ≤ _ := h4
        _ = lam * kappa * ‖x - y‖ := by ring
    simpa [layer, Real.norm_eq_abs] using this

/- Step 1: per-layer error of the iterator. The bound |G - g| ≤ ε is assumed only on the
   calibrated domain D (paper eq. iter-conv) and every pre-activation ∑ⱼ Wᵢⱼ hⱼ is assumed to
   lie in D, which is the paper's standing assumption on the HEAT trajectory. -/
theorem layer_approx_error {d : ℕ} (g G : ℝ → ℝ) (W : Fin d → Fin d → ℝ) (D : Set ℝ) (ε : ℝ)
    (hε : 0 ≤ ε) (hG : ∀ x ∈ D, |G x - g x| ≤ ε) (h : Fin d → ℝ)
    (hdom : ∀ i, (∑ j, W i j * h j) ∈ D) :
    ‖layer G W h - layer g W h‖ ≤ ε := by
  refine (pi_norm_le_iff_of_nonneg hε).2 fun i => ?_
  simpa [layer, Real.norm_eq_abs] using hG _ (hdom i)

/- Step 2: per-layer error of the quantizer (act_perturb row-wise) -/
theorem layer_quant_error {d : ℕ} (g : ℝ → ℝ) (W Q : Fin d → Fin d → ℝ)
    (lam s B : ℝ) (hlam : 0 ≤ lam) (hs : 0 ≤ s) (hB : 0 ≤ B)
    (hg : ∀ x y : ℝ, |g x - g y| ≤ lam * |x - y|)
    (hWQ : ∀ i j, |W i j - Q i j| ≤ s / 2)
    (h : Fin d → ℝ) (hh : ‖h‖ ≤ B) :
    ‖layer g Q h - layer g W h‖ ≤ lam * (s / 2) * d * B := by
  have hh' : ∀ j, |h j| ≤ B := by
    intro j
    have := (norm_le_pi_norm h j).trans hh
    simpa [Real.norm_eq_abs] using this
  refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun i => ?_
  have := act_perturb g lam s B (Q i) (W i) h hlam hs hB hg
    (fun j => by rw [abs_sub_comm]; exact hWQ i j) hh'
  simpa [layer, Real.norm_eq_abs] using this

/- Matched budget: lam * (s/2) * d * B = eps_n at the shifted range -/
theorem matched_step_size (R M r lam B : ℝ) (d : ℕ) (n : ℕ)
    (hR : 0 < R) (hM : 0 < M) (hr0 : 0 < r) (hlam : 0 < lam) (hB : 0 < B) (hd : 0 < d) :
    lam * (step R (matchedBits (lam * d * B * R) M r n) / 2) * d * B = eps M r n := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hpos : 0 < lam * d * B * R := by positivity
  have key := quantizer_error_eq_iterator_error (lam * d * B * R) M r n hpos hM hr0
  rw [← key, step, step]
  ring

/- Step 3: budget required only along the trajectory taken (QAT side) -/
theorem traj_dist_le' (Lex F : ℕ → E → E) (h₀ : E) (ρ ε : ℝ)
    (hρ : 0 ≤ ρ) (hε : 0 ≤ ε)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hpert : ∀ ℓ, ‖F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ)‖ ≤ ε) (L : ℕ) :
    ‖traj F h₀ L - traj Lex h₀ L‖ ≤ ε * ∑ j ∈ range L, ρ ^ j := by
  refine trajectory_bound (fun ℓ => ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖) ρ ε L hρ hε (by simp [traj])
    (fun ℓ _ => ?_) L le_rfl
  show ‖traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)‖ ≤ ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε
  have h1 : traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)
      = (F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)) := by
    simp [traj]
  rw [h1]
  calc ‖(F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ))‖
      ≤ ‖F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ)‖
        + ‖Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)‖ := norm_add_le _ _
    _ ≤ ε + ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ := add_le_add (hpert ℓ) (hlip ℓ _ _)
    _ = ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε := by ring

/- Theorem: matched-precision objective bound (Steps 0–4 assembled).
   Hypotheses follow the appendix: the iterator bound holds on the calibrated domain D and every
   HEAT pre-activation lies in D; the quantizer budget is the paper's b*(n), i.e. the per-layer
   rounding error is at most (not necessarily equal to) ε_n. -/

theorem heat_is_qat_layers {d : ℕ} (f Fn : ℝ → ℝ) (D : Set ℝ) (W Q : ℕ → Fin d → Fin d → ℝ)
    (h₀ : Fin d → ℝ) (loss : (Fin d → ℝ) → ℝ)
    (lam kappa M r ξ s B : ℝ) (n L : ℕ)
    (hlam : 0 ≤ lam) (hkappa : 0 ≤ kappa) (hs : 0 ≤ s) (hB : 0 ≤ B)
    (hξ : 0 ≤ ξ) (hM : 0 ≤ M) (hr : 0 ≤ r)
    (hf : ∀ x y : ℝ, |f x - f y| ≤ lam * |x - y|)
    (hW : ∀ ℓ i, ∑ j, |W ℓ i j| ≤ kappa)
    (hFn : ∀ x ∈ D, |Fn x - f x| ≤ eps M r n)
    (hdom : ∀ ℓ i, (∑ j, W ℓ i j * traj (fun k => layer Fn (W k)) h₀ ℓ j) ∈ D)
    (hQ : ∀ ℓ i j, |W ℓ i j - Q ℓ i j| ≤ s / 2)
    (hbudget : lam * (s / 2) * d * B ≤ eps M r n)
    (hbound : ∀ ℓ, ‖traj (fun k => layer f (Q k)) h₀ ℓ‖ ≤ B)
    (hloss : ∀ x y : Fin d → ℝ, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj (fun k => layer Fn (W k)) h₀ L)
        - loss (traj (fun k => layer f (Q k)) h₀ L)|
      ≤ 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, (lam * kappa) ^ j := by
  set Lex : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer f (W k)
  set FH : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer Fn (W k)
  set FQ : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer f (Q k)
  have hεnn : 0 ≤ eps M r n := mul_nonneg hM (pow_nonneg hr _)
  have hρ : 0 ≤ lam * kappa := mul_nonneg hlam hkappa
  have hlip : ∀ ℓ, ∀ x y : Fin d → ℝ, ‖Lex ℓ x - Lex ℓ y‖ ≤ lam * kappa * ‖x - y‖ :=
    fun ℓ x y => layer_lipschitz f (W ℓ) lam kappa hlam hf (hW ℓ) x y
  have hHb : ∀ ℓ, ‖FH ℓ (traj FH h₀ ℓ) - Lex ℓ (traj FH h₀ ℓ)‖ ≤ eps M r n :=
    fun ℓ => layer_approx_error f Fn (W ℓ) D (eps M r n) hεnn hFn (traj FH h₀ ℓ) (hdom ℓ)
  have hQb : ∀ ℓ, ‖FQ ℓ (traj FQ h₀ ℓ) - Lex ℓ (traj FQ h₀ ℓ)‖ ≤ eps M r n :=
    fun ℓ => (layer_quant_error f (W ℓ) (Q ℓ) lam s B hlam hs hB hf (hQ ℓ)
      (traj FQ h₀ ℓ) (hbound ℓ)).trans hbudget
  have hH' := traj_dist_le' Lex FH h₀ (lam * kappa) (eps M r n) hρ hεnn hlip hHb L
  have hQ' := traj_dist_le' Lex FQ h₀ (lam * kappa) (eps M r n) hρ hεnn hlip hQb L
  have htri : ‖traj FH h₀ L - traj FQ h₀ L‖
      ≤ ‖traj FH h₀ L - traj Lex h₀ L‖ + ‖traj FQ h₀ L - traj Lex h₀ L‖ := by
    rw [← norm_neg (traj FQ h₀ L - traj Lex h₀ L),
      show traj FH h₀ L - traj FQ h₀ L
        = (traj FH h₀ L - traj Lex h₀ L) + -(traj FQ h₀ L - traj Lex h₀ L) by abel]
    exact norm_add_le _ _
  have h2 := hloss (traj FH h₀ L) (traj FQ h₀ L)
  have h3 : ξ * ‖traj FH h₀ L - traj FQ h₀ L‖
      ≤ 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, (lam * kappa) ^ j := by
    calc ξ * ‖traj FH h₀ L - traj FQ h₀ L‖
        ≤ ξ * (eps M r n * (∑ j ∈ range L, (lam * kappa) ^ j)
            + eps M r n * ∑ j ∈ range L, (lam * kappa) ^ j) :=
          mul_le_mul_of_nonneg_left (htri.trans (add_le_add hH' hQ')) hξ
      _ = 2 * ξ * (M * r ^ (2 ^ n)) * ∑ j ∈ range L, (lam * kappa) ^ j := by rw [eps]; ring
  linarith

/- Trajectory bound with a site-dependent budget -/
theorem trajectory_bound_sites (δ : ℕ → ℝ) (ρ : ℝ) (ε : ℕ → ℝ) (L : ℕ)
    (hρ : 0 ≤ ρ) (h0 : δ 0 = 0)
    (hstep : ∀ ℓ, ℓ < L → δ (ℓ + 1) ≤ ρ * δ ℓ + ε ℓ) :
    ∀ ℓ, ℓ ≤ L → δ ℓ ≤ ∑ k ∈ range ℓ, ρ ^ (ℓ - 1 - k) * ε k := by
  intro ℓ
  induction ℓ with
  | zero => intro _; simp [h0]
  | succ m ih =>
      intro hm
      have h1 := hstep m (Nat.lt_of_succ_le hm)
      have h2 := ih (Nat.le_of_succ_le hm)
      have h3 : ρ * δ m ≤ ρ * ∑ k ∈ range m, ρ ^ (m - 1 - k) * ε k :=
        mul_le_mul_of_nonneg_left h2 hρ
      have h4 : ρ * ∑ k ∈ range m, ρ ^ (m - 1 - k) * ε k
          = ∑ k ∈ range m, ρ ^ (m + 1 - 1 - k) * ε k := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun k hk => ?_
        have hk' : k < m := Finset.mem_range.mp hk
        have hidx : m + 1 - 1 - k = (m - 1 - k) + 1 := by omega
        rw [hidx, pow_succ]
        ring
      have h5 : ρ ^ (m + 1 - 1 - m) * ε m = ε m := by
        have hidx : m + 1 - 1 - m = 0 := by omega
        rw [hidx, pow_zero, one_mul]
      rw [Finset.sum_range_succ, h5]
      linarith [h1, h3, h4]

/- With a common budget the per-site sum is the geometric sum of the uniform theorem -/
theorem sum_sites_const (ρ ε : ℝ) (L : ℕ) :
    ∑ k ∈ range L, ρ ^ (L - 1 - k) * ε = ε * ∑ j ∈ range L, ρ ^ j := by
  rw [Finset.sum_range_reflect (fun j : ℕ => ρ ^ j * ε) L, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/- Accumulation across depth with per-site budgets, required only along the trajectory taken -/
theorem traj_dist_le_sites (Lex F : ℕ → E → E) (h₀ : E) (ρ : ℝ) (ε : ℕ → ℝ)
    (hρ : 0 ≤ ρ)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hpert : ∀ ℓ, ‖F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ)‖ ≤ ε ℓ) (L : ℕ) :
    ‖traj F h₀ L - traj Lex h₀ L‖ ≤ ∑ k ∈ range L, ρ ^ (L - 1 - k) * ε k := by
  refine trajectory_bound_sites (fun ℓ => ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖) ρ ε L hρ
    (by simp [traj]) (fun ℓ _ => ?_) L le_rfl
  show ‖traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)‖ ≤ ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε ℓ
  have h1 : traj F h₀ (ℓ + 1) - traj Lex h₀ (ℓ + 1)
      = (F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)) := by
    simp [traj]
  rw [h1]
  calc ‖(F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ))
        + (Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ))‖
      ≤ ‖F ℓ (traj F h₀ ℓ) - Lex ℓ (traj F h₀ ℓ)‖
        + ‖Lex ℓ (traj F h₀ ℓ) - Lex ℓ (traj Lex h₀ ℓ)‖ := norm_add_le _ _
    _ ≤ ε ℓ + ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ := add_le_add (hpert ℓ) (hlip ℓ _ _)
    _ = ρ * ‖traj F h₀ ℓ - traj Lex h₀ ℓ‖ + ε ℓ := by ring

/- Fidelity to the exact network under per-site counts n k (paper eq. fidelity) -/
theorem fidelity_bound_sites (Lex FH : ℕ → E → E) (h₀ : E) (loss : E → ℝ)
    (ρ M r ξ : ℝ) (n : ℕ → ℕ) (L : ℕ)
    (hρ : 0 ≤ ρ) (hξ : 0 ≤ ξ)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hH : ∀ ℓ, ‖FH ℓ (traj FH h₀ ℓ) - Lex ℓ (traj FH h₀ ℓ)‖ ≤ eps M r (n ℓ))
    (hloss : ∀ x y : E, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj FH h₀ L) - loss (traj Lex h₀ L)|
      ≤ ξ * ∑ k ∈ range L, ρ ^ (L - 1 - k) * (M * r ^ (2 ^ (n k))) := by
  have hH' := traj_dist_le_sites Lex FH h₀ ρ (fun k => eps M r (n k)) hρ hlip hH L
  have h2 := hloss (traj FH h₀ L) (traj Lex h₀ L)
  have h3 := mul_le_mul_of_nonneg_left hH' hξ
  simp only [eps] at h3
  linarith

/- Per-site matched precision (paper eq. mixed-precision): HEAT with counts n k against QAT at
   the matched bit widths b*(n k), i.e. per-site rounding budgets eps M r (n k) -/
theorem heat_is_qat_sites (Lex FH FQ : ℕ → E → E) (h₀ : E) (loss : E → ℝ)
    (ρ M r ξ : ℝ) (n : ℕ → ℕ) (L : ℕ)
    (hρ : 0 ≤ ρ) (hξ : 0 ≤ ξ)
    (hlip : ∀ ℓ, ∀ x y : E, ‖Lex ℓ x - Lex ℓ y‖ ≤ ρ * ‖x - y‖)
    (hH : ∀ ℓ, ‖FH ℓ (traj FH h₀ ℓ) - Lex ℓ (traj FH h₀ ℓ)‖ ≤ eps M r (n ℓ))
    (hQ : ∀ ℓ, ‖FQ ℓ (traj FQ h₀ ℓ) - Lex ℓ (traj FQ h₀ ℓ)‖ ≤ eps M r (n ℓ))
    (hloss : ∀ x y : E, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj FH h₀ L) - loss (traj FQ h₀ L)|
      ≤ 2 * ξ * ∑ k ∈ range L, ρ ^ (L - 1 - k) * (M * r ^ (2 ^ (n k))) := by
  have hH' := traj_dist_le_sites Lex FH h₀ ρ (fun k => eps M r (n k)) hρ hlip hH L
  have hQ' := traj_dist_le_sites Lex FQ h₀ ρ (fun k => eps M r (n k)) hρ hlip hQ L
  have htri : ‖traj FH h₀ L - traj FQ h₀ L‖
      ≤ ‖traj FH h₀ L - traj Lex h₀ L‖ + ‖traj FQ h₀ L - traj Lex h₀ L‖ := by
    rw [← norm_neg (traj FQ h₀ L - traj Lex h₀ L),
      show traj FH h₀ L - traj FQ h₀ L
        = (traj FH h₀ L - traj Lex h₀ L) + -(traj FQ h₀ L - traj Lex h₀ L) by abel]
    exact norm_add_le _ _
  have h2 := hloss (traj FH h₀ L) (traj FQ h₀ L)
  have h3 := mul_le_mul_of_nonneg_left (htri.trans (add_le_add hH' hQ')) hξ
  simp only [eps] at h3
  linarith

/- Paper eq. fidelity for the concrete layer h ↦ (g(∑ⱼ Wᵢⱼ hⱼ))ᵢ: the deployed circuit with
   per-site counts n k against the exact network -/
theorem fidelity_bound_layers_sites {d : ℕ} (f : ℝ → ℝ) (Fn : ℕ → ℝ → ℝ) (D : Set ℝ)
    (W : ℕ → Fin d → Fin d → ℝ) (h₀ : Fin d → ℝ) (loss : (Fin d → ℝ) → ℝ)
    (lam kappa M r ξ : ℝ) (n : ℕ → ℕ) (L : ℕ)
    (hlam : 0 ≤ lam) (hkappa : 0 ≤ kappa) (hξ : 0 ≤ ξ) (hM : 0 ≤ M) (hr : 0 ≤ r)
    (hf : ∀ x y : ℝ, |f x - f y| ≤ lam * |x - y|)
    (hW : ∀ ℓ i, ∑ j, |W ℓ i j| ≤ kappa)
    (hFn : ∀ ℓ, ∀ x ∈ D, |Fn ℓ x - f x| ≤ eps M r (n ℓ))
    (hdom : ∀ ℓ i, (∑ j, W ℓ i j * traj (fun k => layer (Fn k) (W k)) h₀ ℓ j) ∈ D)
    (hloss : ∀ x y : Fin d → ℝ, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj (fun k => layer (Fn k) (W k)) h₀ L)
        - loss (traj (fun k => layer f (W k)) h₀ L)|
      ≤ ξ * ∑ k ∈ range L, (lam * kappa) ^ (L - 1 - k) * (M * r ^ (2 ^ (n k))) := by
  set Lex : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer f (W k)
  set FH : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer (Fn k) (W k)
  have hεnn : ∀ ℓ, 0 ≤ eps M r (n ℓ) := fun ℓ => mul_nonneg hM (pow_nonneg hr _)
  have hρ : 0 ≤ lam * kappa := mul_nonneg hlam hkappa
  have hlip : ∀ ℓ, ∀ x y : Fin d → ℝ, ‖Lex ℓ x - Lex ℓ y‖ ≤ lam * kappa * ‖x - y‖ :=
    fun ℓ x y => layer_lipschitz f (W ℓ) lam kappa hlam hf (hW ℓ) x y
  have hHb : ∀ ℓ, ‖FH ℓ (traj FH h₀ ℓ) - Lex ℓ (traj FH h₀ ℓ)‖ ≤ eps M r (n ℓ) :=
    fun ℓ => layer_approx_error f (Fn ℓ) (W ℓ) D (eps M r (n ℓ)) (hεnn ℓ) (hFn ℓ)
      (traj FH h₀ ℓ) (hdom ℓ)
  exact fidelity_bound_sites Lex FH h₀ loss (lam * kappa) M r ξ n L hρ hξ hlip hHb hloss

/- Paper eq. mixed-precision for the concrete layer: per-site counts n k against per-site
   matched quantization (step s k with λ (s k / 2) d B ≤ ε_{n k}) -/
theorem heat_is_qat_layers_sites {d : ℕ} (f : ℝ → ℝ) (Fn : ℕ → ℝ → ℝ) (D : Set ℝ)
    (W Q : ℕ → Fin d → Fin d → ℝ) (h₀ : Fin d → ℝ) (loss : (Fin d → ℝ) → ℝ)
    (lam kappa M r ξ B : ℝ) (s : ℕ → ℝ) (n : ℕ → ℕ) (L : ℕ)
    (hlam : 0 ≤ lam) (hkappa : 0 ≤ kappa) (hs : ∀ ℓ, 0 ≤ s ℓ) (hB : 0 ≤ B)
    (hξ : 0 ≤ ξ) (hM : 0 ≤ M) (hr : 0 ≤ r)
    (hf : ∀ x y : ℝ, |f x - f y| ≤ lam * |x - y|)
    (hW : ∀ ℓ i, ∑ j, |W ℓ i j| ≤ kappa)
    (hFn : ∀ ℓ, ∀ x ∈ D, |Fn ℓ x - f x| ≤ eps M r (n ℓ))
    (hdom : ∀ ℓ i, (∑ j, W ℓ i j * traj (fun k => layer (Fn k) (W k)) h₀ ℓ j) ∈ D)
    (hQ : ∀ ℓ i j, |W ℓ i j - Q ℓ i j| ≤ s ℓ / 2)
    (hbudget : ∀ ℓ, lam * (s ℓ / 2) * d * B ≤ eps M r (n ℓ))
    (hbound : ∀ ℓ, ‖traj (fun k => layer f (Q k)) h₀ ℓ‖ ≤ B)
    (hloss : ∀ x y : Fin d → ℝ, |loss x - loss y| ≤ ξ * ‖x - y‖) :
    |loss (traj (fun k => layer (Fn k) (W k)) h₀ L)
        - loss (traj (fun k => layer f (Q k)) h₀ L)|
      ≤ 2 * ξ * ∑ k ∈ range L, (lam * kappa) ^ (L - 1 - k) * (M * r ^ (2 ^ (n k))) := by
  set Lex : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer f (W k)
  set FH : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer (Fn k) (W k)
  set FQ : ℕ → (Fin d → ℝ) → (Fin d → ℝ) := fun k => layer f (Q k)
  have hεnn : ∀ ℓ, 0 ≤ eps M r (n ℓ) := fun ℓ => mul_nonneg hM (pow_nonneg hr _)
  have hρ : 0 ≤ lam * kappa := mul_nonneg hlam hkappa
  have hlip : ∀ ℓ, ∀ x y : Fin d → ℝ, ‖Lex ℓ x - Lex ℓ y‖ ≤ lam * kappa * ‖x - y‖ :=
    fun ℓ x y => layer_lipschitz f (W ℓ) lam kappa hlam hf (hW ℓ) x y
  have hHb : ∀ ℓ, ‖FH ℓ (traj FH h₀ ℓ) - Lex ℓ (traj FH h₀ ℓ)‖ ≤ eps M r (n ℓ) :=
    fun ℓ => layer_approx_error f (Fn ℓ) (W ℓ) D (eps M r (n ℓ)) (hεnn ℓ) (hFn ℓ)
      (traj FH h₀ ℓ) (hdom ℓ)
  have hQb : ∀ ℓ, ‖FQ ℓ (traj FQ h₀ ℓ) - Lex ℓ (traj FQ h₀ ℓ)‖ ≤ eps M r (n ℓ) :=
    fun ℓ => (layer_quant_error f (W ℓ) (Q ℓ) lam (s ℓ) B hlam (hs ℓ) hB hf (hQ ℓ)
      (traj FQ h₀ ℓ) (hbound ℓ)).trans (hbudget ℓ)
  exact heat_is_qat_sites Lex FH FQ h₀ loss (lam * kappa) M r ξ n L hρ hξ hlip hHb hQb hloss

end HEEquivQAT
