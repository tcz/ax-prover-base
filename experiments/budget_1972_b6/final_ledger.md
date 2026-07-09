## STRATEGY LEDGER — Putnam 1972 B6

### Strategies Attempted

**1. SIMPLE GEOMETRIC SERIES BOUND (|z|/(1−|z|) < 1)**
- Runs: 9
- Verdict: **DEAD END**
- Blocker: Requires |z| < 0.5; cannot reach φ ≈ 0.618.

**2. (2−z)/(1−z) IDENTITY (Evan Chen variant)**
- Runs: 8
- Verdict: **DEAD END**
- Blocker: Only yields |z| ≥ 0.5. Inequality direction doesn't close for φ.

**3. CASE SPLIT ON MINIMUM EXPONENT (n_i ≥ i+1)**
- Runs: 13
- Verdict: **DEAD END**
- Blocker: Triangle inequality cannot reach φ regardless of exponent constraints.

**4. CASE SPLIT ON k WITH TRIANGLE INEQUALITY**
- Runs: 75 (+12 this run)
- Verdict: **DEAD END**
- Blocker: φ + φ² = 1 exactly. For r ∈ [0.5, φ) the comparison r^{k+1} vs 2r−1 fails (e.g., r=0.6, k=3 gives 0.13 < 0.2). No finite-k triangle bound closes the gap to φ.

**5. BOUNDED TERM COUNT / REVERSE TRIANGLE**
- Runs: 2
- Verdict: **DEAD END**

### Critical Realization (CONFIRMED — 12 RUNS)

**All magnitude-based approaches use |∑z^{n_i}| ≤ ∑|z|^{n_i}, discarding phase. This provably cannot yield |z| ≥ φ.** The golden ratio emerges from φ² + φ = 1, an algebraic identity requiring structure beyond norm bounds. No triangle-inequality variant should be attempted again.

### Untried Directions

1. **Rouché's theorem**: On |z| = r < φ, show |∑z^{n_i}| < |1| = 1. Need phase-aware bound exploiting strictly increasing exponents—angular cancellation may give sharper bound than ∑|z|^{n_i}.

2. **Real-part analysis**: From ∑z^{n_i} = −1, deduce Re(∑z^{n_i}) = −1. For |z| < φ, show Re cannot reach −1 by analyzing cos(n_i·arg(z)) contributions with gap constraints.

3. **Fibonacci/Zeckendorf connection**: φ satisfies x² + x = 1. Strictly increasing exponents exclude consecutive-term sums in a Zeckendorf sense. Explore whether this structural constraint bounds achievable sums.

4. **Polynomial factorization**: For fixed small k, the polynomial 1 + z^{n₁} + ... + z^{nₖ} may factor or have known root locations. Check if roots lie outside B(0,φ) for all valid exponent choices.
