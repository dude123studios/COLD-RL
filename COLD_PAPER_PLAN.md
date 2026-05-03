╔══════════════════════════════════════════════════════════════════════════════╗
║       COLD-RL: FINAL EXPERIMENT PLAN                                        ║
║       A100 SXM 80GB  ×  2 (no separate eval GPU)                            ║
║       Goal: beat DARLING + ModC + Power Sampling at ALL pass@k              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Last updated: 2026-05-03
Status: living document — tick off runs as they complete

GPU IDENTITY:
  GPU-A (this machine) = A100 SXM 80GB — runs the A-track (Qwen3-4B + ablations)
  GPU-G (second machine) = A100 SXM 80GB — runs the G-track (Qwen2.5-7B + 3 seeds)
  Both machines are identical hardware.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 0: HARDWARE REALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Both GPUs — A100 SXM (80GB):
    GPU:    Ampere, 80GB HBM2e, 2.0 TB/s bandwidth
    Extras: Flash Attention 2, BF16 training
    Fits:   Qwen2.5-7B LoRA (~55GB peak), Qwen3-4B LoRA (~38GB peak)
    No fit: Qwen3-14B LoRA (~115GB) — no 14B runs on either machine

  TRAINING TIME ESTIMATES (per GPU):
    Model             | A100 SXM 80GB
    ------------------+--------------
    Qwen2.5-7B  GRPO  |    20h
    Qwen2.5-7B  LoRA  |    20h
    Qwen3-4B    LoRA  |    18h
    DARLING   (Qw3-4B)|    22h

  EVAL (runs in explicit windows after each training run, same GPU):
    Qwen2.5-7B: 500 problems x 200 samples x ~300 tokens avg → ~3h per sweep
    Eval uses ~14GB for 7B BF16 inference — fits easily.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: ALL 20 RUNS AT A GLANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  +----+------+--------------------+-----------------------+------+---------+
  | ID | GPU  | Model              | Algorithm             | hrs  | Purpose |
  +----+------+--------------------+-----------------------+------+---------+
  | G1 |GPU-G | Qwen2.5-7B-Base   | Zero-shot diagnostic  |  2h  | GATE    |
  | G2 |GPU-G | Qwen2.5-7B-Base   | GRPO baseline         |  8h  | CRIT.   |
  | G3 |GPU-G | Qwen2.5-7B-Base   | COLD-RL (i,k) seed 1  |  8h  | CRIT.*  |
  | G4 |GPU-G | Qwen2.5-7B-Base   | COLD-RL (i,k) seed 2  |  8h  | CRIT.   |
  | G5 |GPU-G | Qwen2.5-7B-Base   | COLD-RL (i,k) seed 3  |  8h  | CRIT.   |
  | G6 |GPU-G | Qwen2.5-7B-Base   | ModC reproduction     |  7h  | CRIT.   |
  | G7 |GPU-G | Qwen2.5-Math-7B   | COLD-RL (i,k)         |  9h  | HIGH    |
  | G8 |GPU-G | Qwen2.5-7B-Base   | GRPO 12k steps        | 12h  | HIGH    |
  | G9 |GPU-G | Qwen2.5-7B-Base   | k-schedule sparse abl.|  8h  | MEDIUM  |
  |G10 |GPU-G | --                | BUFFER / help A100    |  --  | FLEX    |
  +----+------+--------------------+-----------------------+------+---------+
  | A1 | A100 | Qwen3-4B-Base     | GRPO baseline         | 18h  | CRIT.   |
  | A2 | A100 | Qwen3-4B-Base     | DARLING reproduction  | 22h  | CRIT.   |
  | A3 | A100 | Qwen3-4B-Base     | COLD-RL (i,k)         | 22h  | CRIT.   |
  | A4 | A100 | Qwen2.5-7B-Base   | Ablation: soft token  | 20h  | CRIT.abl|
  | A5 | A100 | Qwen2.5-7B-Base   | Ablation: k-only pfx  | 20h  | CRIT.abl|
  | A6 | A100 | Qwen2.5-7B-Base   | Ablation: no Phase 1  | 20h  | HIGH    |
  | A7 | A100 | Qwen2.5-7B-Base   | Ablation: fixed lam   | 20h  | HIGH    |
  | A8 | A100 | Qwen2.5-7B-Base   | Ablation: lexical r_d | 20h  | HIGH    |
  | A9 | A100 | Qwen2.5-7B-Base   | Puri et al. multi-ans | 22h  | HIGH    |
  |A10 | A100 | Qwen2.5-7B-Base   | Power Sampling infer. |  4h  | FREE    |
  +----+------+--------------------+-----------------------+------+---------+

  * G3 = most important single run in the plan

  FREE ANALYSES (no training, scheduled in eval windows):
    F1: Bootstrap CIs            F2: Inference prefix on G2   F3: Faithfulness judge
    F4: Training dynamics        F5: Temperature sweep        F6: Self-consistency
    F7: Best-of-N with PRM       F8: DARLING classifier prep  F9: t-SNE + examples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: RUN SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- G1: ZERO-SHOT DIAGNOSTIC (GPU-G, 2h, inference only) ---------------------
  PURPOSE: Confirm "[PARALLEL SAMPLE i OF k]" prefix induces diversity in
           base model before any training. GO/NO-GO gate for G3.
  MODEL:   Qwen/Qwen2.5-7B (raw, no fine-tuning)
  PROTOCOL:
    100 problems from MATH-500, stratified by difficulty
    Arm A: k=8 samples with "[PARALLEL SAMPLE {i} OF 8]" for i=1..8, tau=0.7
    Arm B: k=8 i.i.d. samples, no prefix, tau=0.7
    Arm C: k=8 i.i.d. samples, no prefix, tau=1.1 (temperature baseline)
    Measure: mean pairwise E5-large-instruct cosine distance, pass@8
  GO:    Arm A semantic distance > Arm B by >0.04 AND pass@8 > Arm B
  NO-GO: Arm A ~= Arm B -> revise prefix wording and re-run (2h max):
         "[PARALLEL SAMPLE {i} OF {k}: use a different solution strategy
          than other parallel samples would use for this problem type]"
  CMD:   python scripts/zero_shot_diagnostic.py \
           --model Qwen/Qwen2.5-7B --n_problems 100 --k 8

-- G2: GRPO BASELINE (GPU-G, 8h train + 2h eval = 10h) ----------------------
  MODEL:   Qwen/Qwen2.5-7B
  DATA:    open-thoughts/OpenThoughts-114k + MATH train split
  ALGO:    GRPO, lambda_div=0.0
  STEPS:   8,000  |  BATCH: 64 prompts x 8 rollouts
  LR:      1e-6 cosine with 50-step warmup  |  KL beta: 0.04  |  Clip: 0.2
  SAVE:    every 2,000 steps (4 checkpoints)
  EVAL:    MATH-500, GPQA-Diamond, HumanEval+, NuminaMath-500
           pass@k for k in {1,2,4,8,16,32,64,128}, n=200 per problem
  POWER SAMPLING: Apply A10 to this checkpoint immediately after eval.

-- G3/G4/G5: COLD-RL SEEDS 1/2/3 (GPU-G, 8h + 2h eval each) * --------------
  SEEDS:    42 (G3), 1337 (G4), 0 (G5)
  MODEL:    Qwen/Qwen2.5-7B -> LoRA rank=32, alpha=64
  DATA:     Same as G2

  PREFIX FORMAT (exact string, no variation):
    "[PARALLEL SAMPLE {i} OF {k}]"
    Prepended as text before the problem statement.
    k=1,i=1: "[PARALLEL SAMPLE 1 OF 1]" -> quality-only, no diversity role
    k=8,i=5: "[PARALLEL SAMPLE 5 OF 8]" -> 5th of 8 independent solvers
    NEVER include explicit mode labels. Roles learned from reward alone.

  REWARD:
    R(y_i, S_k) = r_q(y_i) + lambda(k) * r_d(y_i, S_k \ {y_i})
    r_q = binary correctness (0 or 1 from math verifier)
    r_d = mean cosine distance from E5-large-instruct embeddings of the
          OTHER k-1 samples. STOP-GRADIENT through encoder always.
    lambda(k) = alpha * log(k) / log(16),  alpha = 0.3
    lambda(1) = 0.0 EXACTLY — asserted in code: assert abs(lambda_fn(1)) < 1e-9

  TRAINING SCHEDULE:
    Phase 1 (steps 1-4,000):    k=1, i=1 only. Pure quality. lambda=0.
    Phase 2 (steps 4,001-8,000): k ~ Uniform{1,2,4,8,16} per batch.
                                  i ~ Uniform{1,...,k} per sample.
    Within each batch: all k samples for a given prompt generated simultaneously.
    GRPO group-relative advantage normalization within the k-group.

  LORA:    rank=32, alpha=64
           targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  LR:      5e-5 (LoRA), 0 (frozen base)
  KL beta: 0.04  |  Steps: 8,000  |  Save: every 1,000 steps
  tau:     1.0 during training, 0.7 at eval

  F4 DYNAMICS LOGGING (add before G3 starts, zero overhead):
    Every 500 steps: r_q (k=1), r_d (k=8), pairwise dist for k in {1,4,8,16},
    pass@1 and pass@8 on 50-problem val subset. -> Figure 2 in paper.

-- G6: ModC REPRODUCTION (GPU-G, 7h train + 2h eval = 9h) ------------------
  MODEL:   Qwen/Qwen2.5-7B-Base  (SAME as G2/G3 — fair comparison)
  DATA:    open-thoughts/OpenThoughts-114k
  ALGO:    ModC with mode-specific prefix SFT distillation
           Teachers: Qwen2.5-72B-Instruct (CoT) and (verification-step prompt)
           Prefix A: "[REASONING: direct solution]"
           Prefix B: "[REASONING: step-by-step verification]"
           AdamW lr=2e-5, cosine, 4 epochs — NOT RL, SFT loss
  VERIFY:  Reproduce their Table 1 pass@k within 2pp. Disclose any deviation.

-- G7: COLD-RL Qwen2.5-Math-7B (GPU-G, 9h + 2h eval) -----------------------
  MODEL:   Qwen/Qwen2.5-Math-7B
  DATA:    MATH training split (Power Sampling's exact data)
  ALGO:    Identical to G3 (same prefix, reward, curriculum)
  PURPOSE: Direct Power Sampling comparison on their exact model + data.

-- G8: GRPO 12k STEPS (GPU-G, 12h + 2h eval) --------------------------------
  MODEL:   Qwen/Qwen2.5-7B-Base  |  ALGO: GRPO  |  STEPS: 12,000
  PURPOSE: Rules out "COLD-RL just benefits from more compute."
           If GRPO@12k ~= GRPO@8k: extra training doesn't explain gains.
  EVAL:    MATH-500 only (robustness check, not full eval)

-- G9: k-SCHEDULE SPARSE ABLATION (GPU-G, 8h + 1h eval) --------------------
  MODEL:   Qwen/Qwen2.5-7B-Base
  ALGO:    G3 but Phase 2 k schedule = {1, 4, 16} (skip k=2 and k=8)
  PURPOSE: Does model need dense k coverage? Test generalization to skipped k.

-- G10: BUFFER ---------------------------------------------------------------
  Default: GRPO baseline for Qwen2.5-Math-7B (~8h) needed as G7 reference.
  Fallback: pick up A7 or A8 from A100 queue to speed up ablations.

-- A1: GRPO Qwen3-4B BASELINE (A100, 18h + 3h eval) -------------------------
  MODEL:   Qwen/Qwen3-4B (base)
  DATA:    agentica-org/DeepScaleR-Preview-Dataset
  ALGO:    GRPO, lambda_div=0.0  |  STEPS: 8,000  |  BATCH: 64 x 8
  LR:      2e-6  |  KL beta: 0.04  |  Clip: 0.2
  EVAL:    AIME25, OlympiadBench, MATH-500
           pass@k for k in {1,2,4,8,16,64,128}  (k=128 to match DARLING)
  PURPOSE: DARLING comparison baseline. DARLING uses exactly this model+data.

-- A2: DARLING REPRODUCTION (A100, 22h + 3h eval) ---------------------------
  MODEL:   Qwen/Qwen3-4B  |  DATA: DeepscaleR
  PREREQ:  F8 classifier training runs in A1 eval window (4h on A100).
  ALGO:    DARLING (arXiv:2509.02534)
           r_total = r_q x (1 + gamma * r_diversity)
           r_diversity from semantic equivalence classifier (F8)
           gamma: their reported appendix value
  NOTE:    If F8 classifier quality insufficient, fallback to cosine distance
           from sentence-transformers/all-MiniLM-L6-v2. Flag in paper.

-- A3: COLD-RL Qwen3-4B (A100, 22h + 3h eval) -------------------------------
  MODEL:   Qwen/Qwen3-4B  |  DATA: DeepscaleR
  ALGO:    Identical to G3 (same prefix, reward, curriculum — must match exactly)
  PURPOSE: Beat DARLING on their exact model, data, and benchmarks.
  EVAL:    AIME25, OlympiadBench, MATH-500, pass@k k in {1,2,4,8,16,64,128}

-- A4: SOFT-TOKEN ABLATION (A100, 20h + 2h eval) ----------------------------
  ALGO:    Replace (i,k) text prefix with soft token e(k) = f_psi(log k),
           2-layer MLP (256 hidden), projected to model dim.
           ALL k samples get SAME embedding. No per-sample role index i.
  PURPOSE: Proves i component (role assignment) is critical, not just k.

-- A5: k-ONLY PREFIX ABLATION (A100, 20h + 2h eval) -------------------------
  ALGO:    Text prefix "[GROUP SIZE: {k}]" — all k samples see same prefix.
           No role index i. Same reward as G3.
  PURPOSE: Text-communicated k without role index. Should show G3 >> A5.

-- A6: NO PHASE-1 CURRICULUM (A100, 20h + 2h eval) --------------------------
  ALGO:    G3 but k ~ Uniform{1,2,4,8,16} from step 1. No quality warmup.
  EXPECTED: pass@1 drops 3-5pp. Confirms two-phase is necessary.

-- A7: FIXED LAMBDA ABLATION (A100, 20h + 2h eval) --------------------------
  ALGO:    G3 but lambda = 0.3 for ALL k. Directly mimics DARLING's design.
  EXPECTED: pass@1 degrades (diversity pressure always on), pass@k smaller gains.

-- A8: LEXICAL DIVERSITY REWARD (A100, 20h + 2h eval) -----------------------
  ALGO:    G3 but r_d = complement of avg BLEU-4 vs other k-1 responses.
  EXPECTED: Lower gains on hard problems where paraphrasing is rewarded.

-- A9: PURI et al. MULTI-ANSWER RL (A100, 22h + 2h eval) --------------------
  MODEL:   Qwen/Qwen2.5-7B-Base  |  DATA: OpenThoughts
  ALGO:    K=8 answers in ONE forward pass, set-level correctness reward.
           "[Answer 1]: ... [Answer 2]: ... [Answer 8]: ..."
  PURPOSE: Beat at k != 8. Show COLD-RL wins at k=1 (quality mode) and k=16+.

-- A10: POWER SAMPLING (A100, 4h inference, FREE) ----------------------------
  TRAINING: NONE. Inference from G2 checkpoint.
  ALGO:    Power Sampling MCMC (ICLR 2026). alpha=4.0, Tmax=3072, block=192.
  EVAL:    MATH-500, GPQA, HumanEval+, pass@k k in {1,2,4,8,16,64}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: FREE ANALYSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  F1: Bootstrap 95% CIs (CPU, ~2h, mandatory for all table numbers)
  F2: Inference prefix on G2 GRPO ckpt (~4h) — kills "just prompt engineering"
  F3: Faithfulness judge via Qwen3-72B (~4h) — shows diverse strategies
  F4: Training dynamics logging (zero overhead, hooks added before G3)
  F5: Temperature sweep from G2 outputs (~3h) — GRPO at tau in {0.6..1.2}
  F6: Self-consistency majority vote from G2 outputs (~1h CPU)
  F7: Best-of-N with Math-Shepherd PRM from G2 outputs (~3h)
  F8: DARLING classifier pretraining (A100, ~4h, runs during A1 eval window)
  F9: t-SNE + qualitative examples from G3 outputs (~2h CPU)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: GANTT (both GPUs, explicit eval windows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Hours ->  0    12   24   36   48   60   72   84   96
            |    |    |    |    |    |    |    |    |

  GPU-G    [G1][--G2--][ev+F5+F2][--G3s1--][ev+F3][--G3s2--][ev][--G3s3--]
  A100     [--------A1: GRPO 4B--------][F8][-------A2: DARLING-----------]

  Hours ->  96  108  120  132  144  156  168  180  192
            |    |    |    |    |    |    |    |    |

  GPU-G    [ev][--G6:ModC--][ev][--G7:Math7B--][ev][G8:12k][ev][G9:ksched]
  A100     [ev][-------A3: COLD-RL 4B----------][ev][A4:SoftTok][ev][A5]

  KEY MILESTONES:
    Hour 2:   G1 GO/NO-GO decision on (i,k) prefix
    Hour 43:  3 COLD-RL seeds done on GPU-G -> mean+/-std ready
    Hour 47:  A2 DARLING done -> DARLING comparison baseline ready
    Hour 69:  A3 COLD-RL 4B done -> DARLING beaten at all k
    Hour 86:  GPU-G primary queue done -> all main results available
    Hour 94:  *** START WRITING THE PAPER ***
    Hour ~170: All ablations complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5: COMPARISON TABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TABLE 1: Qwen2.5-7B vs ModC + Power Sampling (MATH-500, GPQA, HumanEval+)
    Rows: GRPO tau-sweep (F5), SC (F6), BoN-PRM (F7), GRPO+prefix (F2),
          Power Sampling (A10), ModC (G6), Puri et al. (A9),
          COLD-RL mean+/-std (G3+G4+G5 + F1 CIs)
    Cols: pass@k for k in {1,2,4,8,16,64}
    Claim: COLD-RL Pareto-dominates all rows.

  TABLE 2: Qwen3-4B vs DARLING (AIME25, OlympiadBench, MATH-500)
    Rows: GRPO (A1), DARLING (A2), COLD-RL (A3)
    Cols: pass@k for k in {1,2,4,8,16,64,128}
    Claim: COLD-RL > DARLING at every k including pass@1 AND pass@128.

  TABLE 3: Power Sampling comparison (Qwen2.5-Math-7B, MATH-500, GPQA)
    Rows: GRPO Math-7B (G10), Power Sampling (A10 on G10), COLD-RL (G7)
    Cols: pass@k for k in {1,2,4,8,16,64}

  TABLE 4: Ablations (Qwen2.5-7B, MATH-500, pass@1 / pass@16)
    Rows: COLD-RL full (G3) | k-only (A5) | soft-token (A4) |
          no Phase-1 (A6) | fixed lambda (A7) | lexical r_d (A8) | GRPO+prefix (F2)

  TABLE 5: Statistical validity — all Table 1-3 numbers with 95% CI (F1)
  FIGURE 2: Training dynamics Phase 1->2 (F4)
  FIGURE 3: t-SNE of parallel samples (F9)
  FIGURE 4: Pareto frontier pass@1 vs pass@16 all methods

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6: REVIEWER ATTACK -> DEFENSE MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "It's just prompt engineering"       -> F2 (GRPO+prefix, no training)
  "No confidence intervals"            -> F1 (bootstrap CIs on all numbers)
  "No multi-seed"                      -> G3+G4+G5 (3 seeds)
  "More training explains the gains"   -> G8 (GRPO 12k steps)
  "Your ModC repro is wrong"           -> G6 vs paper, disclose delta
  "No statistical significance"        -> F1 + F3 p-values
  "Did you test out-of-domain?"        -> Table 1 GPQA + HumanEval+
  "Temperature baseline missing"       -> F5 (tau sweep)
  "Majority vote works just as well"   -> F6 (SC row in Table 1)
  "Best-of-N with PRM works better"    -> F7 (BoN row in Table 1)
  "Model actually using diff methods?" -> F3 (faithfulness judge)
  "Fixed lambda = DARLING in disguise" -> A7 (shows gap)
  "Soft token works just as well"      -> A4 (shows it doesn't)
  "Phase 1/2 ratio arbitrary"          -> F4 dynamics figure
  "Semantic metric vs lexical"         -> A8 (lexical ablation)
  "k schedule matters"                 -> G9 (sparse k)
  "Results only on Qwen family"        -> A1-A3 add Qwen3-4B as 2nd family
  "Qualitative examples missing"       -> F9 (t-SNE + examples)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7: SHARED HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  optimizer:          AdamW
  lr_schedule:        cosine with 100-step linear warmup
  weight_decay:       0.01
  grad_clip:          1.0
  kl_coeff beta:      0.04
  clip_ratio eps:     0.2
  LoRA rank:          32 (alpha=64)
  LoRA targets:       q,k,v,o,gate,up,down projections
  max_seq_len:        8192
  training tau:       1.0
  eval tau:           0.7 (fixed for ALL methods — fair comparison)
  n_eval_samples:     200 per problem (unbiased Chen et al. 2021 estimator)
  encoder:            intfloat/e5-large-instruct (frozen, BF16, stop-gradient)

  COLD-RL specific:
    alpha:            0.3 (lambda multiplier)
    lambda(k):        alpha * log(k) / log(k_max),  k_max=16
    lambda(1):        0.0 (hard zero, asserted in code)
    Phase 1 steps:    4,000 (k=1 only)
    Phase 2 steps:    4,000 (k ~ Uniform{1,2,4,8,16})
    save_steps:       1,000 (8 checkpoints for dynamics analysis)
    k_max_eval:       128 (generalization to unseen k)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 8: RISKS AND MITIGATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RISK 1: G1 NO-GO (prefix ignored)
  ACTION:  Revise prefix to explicit instruction version. Rerun G1 (2h). Delay G2 by 2h.

  RISK 2: G3 fails to improve pass@1 vs G2
  ACTION:  Increase Phase 1 steps from 4000 to 5500 using G10 buffer slot.

  RISK 3: DARLING reproduction deviates >3pp
  ACTION:  Report deviation explicitly. Comparison against reproduced baseline is valid.

  RISK 4: A100 ablations exceed time budget
  ACTION:  GPU-G picks up A7 and A8 after hour 86 (G10 buffer). Priority: A4 > A5 > A7 > A8.

  RISK 5: COLD-RL loses to ModC at pass@1
  ACTION:  Extend Phase 1 to 5500 steps. If still losing at pass@1, report honestly.
           The core claim is pass@k for k>1. pass@1 >= GRPO is sufficient.
