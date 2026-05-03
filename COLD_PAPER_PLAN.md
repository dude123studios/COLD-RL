╔══════════════════════════════════════════════════════════════════════════════╗
║          COLD-RL: 20-RUN EXPERIMENT PLAN + 3-GPU SCHEDULING                ║
║    Goal: beat DARLING + ModC + Power Sampling at pass@k for every k        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Last updated: 2026-05-03
Status: living document — tick off runs as they complete

GPU IDENTITY (THIS MACHINE):
  GPU C = THIS workspace machine = A100-SXM4-80GB (80GB)
  GPU A = 24GB consumer/workstation GPU (to be provisioned)
  GPU B = 96GB server GPU (H100 NVL or dual-A100 NVLink, to be provisioned)

CURRENT STATUS:
  R01 is running on GPU C but with WRONG model (Qwen2.5-7B-Instruct).
  Must kill and restart with Qwen/Qwen2.5-7B (base). See SETUP.md §2.
  R15 should be the first thing run on GPU A once provisioned.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 0: COMPETITOR AUDIT — WHY TWO TRACKS ARE REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  COMPETITOR         BASE MODEL          DATA              WHAT THEY BEAT AT
  ─────────────────────────────────────────────────────────────────────────────
  ModC (ICLR 2026)   Qwen2.5-7B-Base    OpenThoughts,     pass@k via fixed
  arXiv:2512.01127   OLMo2-Base (0.5B   NuminaMath        mode-prefix distill
                     to 7B)                                4x efficiency gain
  ─────────────────────────────────────────────────────────────────────────────
  DARLING            Qwen3-4B-Base,     DeepscaleR        pass@1 AND pass@128
  arXiv:2509.02534   Qwen3-14B-Base                       via multiplicative
                                                          quality x diversity RL
  ─────────────────────────────────────────────────────────────────────────────
  Power Sampling     Qwen2.5-Math-7B,   MATH training     pass@k via MCMC
  ICLR 2026          Qwen2.5-7B,        split             inference, no train
                     Phi-3.5-mini
  ─────────────────────────────────────────────────────────────────────────────
  DivPO              Llama-3.1-8B-Inst  persona/story     story diversity --
  arXiv:2501.18101   (NOT reasoning)    tasks             NOT pass@k math
                                                          -> cite, don't repro
  ─────────────────────────────────────────────────────────────────────────────

  OVERLAP:  Qwen2.5-7B appears in BOTH ModC and Power Sampling.
            -> Primary track:   Qwen2.5-7B-Base beats ModC + Power Sampling
            -> Secondary track: Qwen3-4B-Base beats DARLING
            -> Scale:           Qwen3-14B-Base beats DARLING at 14B

  WHY COLD-RL SHOULD WIN AT EVERY k:
    ModC:   allocates fixed predefined modes uniformly regardless of k.
            At k=1 it may pick a suboptimal mode. At k=2 it still assigns
            two fixed modes even if only one is needed. It cannot condition
            on k at all -- there is no mechanism to be pass@1-optimal
            and pass@16-optimal from the same model.
    DARLING: unconditional diversity. lambda is fixed regardless of k. At k=1,
            diversity pressure is still applied, degrading single-sample
            quality. Cannot shut off diversity for k=1 deployment.
    Power Sampling: inference-only, cannot train diversity in. Fixed
            sharpening/flattening of base distribution -- no k control.

  COLD-RL wins because it is the ONLY method that conditions on k,
  making it Pareto-optimal at every k from a single trained model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: GPU CAPABILITY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GPU A: 24GB  (consumer/workstation -- RTX 4090 or A5000)
  GPU B: 96GB  (server -- H100 NVL 94GB or dual-A100 NVLink)
  GPU C: 80GB  (server -- A100-SXM 80GB)  <- THIS MACHINE

  WHAT FITS WHERE:

  MODEL              OPERATION        24GB    80GB    96GB
  ----------------------------------------------------------
  Qwen2.5-7B         TRAINING (LoRA)  NO(1)   YES(2)  YES
  Qwen2.5-7B         INFERENCE (bf16) NO(3)   YES     YES
  Qwen2.5-7B         INFERENCE (4-bit)YES     YES     YES
  Qwen3-4B           TRAINING (LoRA)  NO(1)   YES(2)  YES
  Qwen3-4B           INFERENCE (bf16) YES(4)  YES     YES
  Qwen3-14B          TRAINING (LoRA)  NO      NO(5)   YES(6)
  Qwen3-14B          INFERENCE (4-bit)NO      YES(7)  YES
  E5-large-instruct  INFERENCE        YES     YES     YES
  Qwen3-Embedding-4B INFERENCE        YES     YES     YES

  (1) 7B LoRA: ~65GB (bf16 weights 14GB + AdamW optimizer 28GB + acts ~23GB)
  (2) 7B LoRA on 80GB: use gradient checkpointing, DeepSpeed ZeRO-1
  (3) 7B bf16: 14GB weights + 8GB KV cache for batch-8 -> 22GB -> marginal NO
  (4) 4B bf16: 8GB weights + 4GB KV -> 12GB -> YES
  (5) 14B LoRA on 80GB: ~115GB needed -> NO
  (6) 14B LoRA on 96GB: use 8-bit optimizer (bitsandbytes) -> ~75GB -> YES (tight)
  (7) 14B 4-bit GGUF/AWQ: ~8GB model -> YES on 80GB for inference

  RULES:
    GPU A (24GB) -> evaluation, inference (4-bit), diagnostics, metrics ONLY
    GPU C (80GB) -> Qwen2.5-7B training + Qwen3-4B training (THIS MACHINE)
    GPU B (96GB) -> Qwen3-14B training + Qwen2.5-7B/3-4B overflow

  VLLM ON 24GB:
    Load checkpoint in 4-bit (AWQ or GPTQ). Use for all eval pass@k runs.
    vLLM with tensor_parallel_size=1, gpu_memory_utilization=0.92
    Throughput: ~200 tokens/s for 7B-4bit -> 200 samples x 500 problems
               x ~300 tokens avg = 30M tokens -> ~41h per full eval sweep
    -> Schedule evals to run CONTINUOUSLY on 24GB in background
    -> Each eval sweep starts immediately after a training checkpoint saves

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: THE 20 RUNS -- ASSIGNED TO GPUs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRACK A: Qwen2.5-7B-Base  -- defeats ModC and Power Sampling
  TRACK B: Qwen3-4B/14B     -- defeats DARLING at both scales
  INFERENCE: 24GB            -- all evaluation, zero-shot, Power Sampling

  +-------------------------------------------------------------------------+
  |  RUN  | GPU | MODEL           | ALGO        |  EST.h |  STATUS          |
  +-------------------------------------------------------------------------+
  |  R01  |  C  | Qwen2.5-7B-Base | GRPO base   |  20h   | RESTARTING*      |
  |  R02  |  B  | Qwen3-14B-Base  | GRPO base   |  40h   | pending          |
  |  R03  |  C  | Qwen2.5-7B-Base | COLD-RL(i,k)|  26h   | pending          |
  |  R04  |  B  | Qwen3-14B-Base  | COLD-RL(i,k)|  54h   | pending          |
  |  R05  |  C  | Qwen2.5-7B-Base | ModC repro  |  22h   | pending          |
  |  R06  |  B  | Qwen3-4B-Base   | DARLING repr|  24h   | pending          |
  |  R07  |  B  | Qwen3-4B-Base   | COLD-RL(i,k)|  24h   | pending          |
  |  R08  |  B  | Qwen3-4B-Base   | GRPO base   |  18h   | pending          |
  |  R09  |  C  | Qwen2.5-7B-Base | Soft-token  |  24h   | pending          |
  |  R10  |  C  | Qwen2.5-7B-Base | k-only pfx  |  24h   | pending          |
  +-------------------------------------------------------------------------+
  |  R11  |  C  | Qwen2.5-7B-Base | No Phase-1  |  24h   | pending          |
  |  R12  |  C  | Qwen2.5-7B-Base | Fixed lam   |  24h   | pending          |
  |  R13  |  C  | Qwen2.5-7B-Base | Lexical r_d |  24h   | pending          |
  |  R14  |  C  | Qwen2.5-7B-Base | Puri et al. |  24h   | pending          |
  |  R15  |  A  | Qwen2.5-7B-Base | Zero-shot   |   2h   | pending          |
  |  R16  |  B  | Qwen2.5-Math-7B | COLD-RL(i,k)|  26h   | pending          |
  +-------------------------------------------------------------------------+
  |  R17  |  A  | Qwen2.5-7B-Base | Power Smp   |   0h   | pending          |
  |  R18  |  C  | Qwen2.5-7B-Base | alpha=0.1   |  24h   | pending          |
  |  R19  |  C  | Qwen2.5-7B-Base | alpha=0.5   |  24h   | pending          |
  |  R20  |  -- | --              | BUFFER      |  24h   | reserved         |
  +-------------------------------------------------------------------------+

  * R01: currently running with wrong model (Instruct). Kill and restart
         with Qwen/Qwen2.5-7B (base). See SETUP.md for exact command.
  * R03 = most important single run in the plan.

  R17 (Power Sampling) is inference-only: applied to R01 checkpoint on 24GB.
      Zero training GPU-hours consumed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: RUN SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- R01: GRPO Qwen2.5-7B-Base (GPU C, 20h) -----------------------------------
  HF model:   Qwen/Qwen2.5-7B
  Data:       OpenThoughts (hf: open-thoughts/OpenThoughts-114k)
              + MATH train split for coverage
  Algorithm:  GRPO, Shao et al. 2025 default hyperparams
  Steps:      8,000  |  Batch: 64 prompts x 8 rollouts
  LR:         1e-6   |  KL beta: 0.04  |  Clip eps: 0.2
  Purpose:    ModC comparison baseline + Power Sampling base checkpoint.
              Power Sampling (R17) is applied directly to this checkpoint.
  Save:       Checkpoint every 2,000 steps. All 4 checkpoints kept.

-- R02: GRPO Qwen3-14B-Base (GPU B, 40h) ------------------------------------
  HF model:   Qwen/Qwen3-14B  (base, NOT instruct)
  Data:       DeepscaleR (hf: agentica-org/DeepScaleR-Preview-Dataset)
  Algorithm:  GRPO, same hyperparams scaled for 14B
  Steps:      8,000  |  Batch: 32 prompts x 8 rollouts (smaller due to size)
  Optimizer:  8-bit AdamW (bitsandbytes) to fit in 96GB
  LR:         5e-7   |  Grad checkpoint: ON
  Purpose:    DARLING 14B comparison baseline.
              CRITICAL: must match DARLING's exact reported training setup.

-- R03: COLD-RL (i,k) Qwen2.5-7B-Base (GPU C, 26h) * MOST IMPORTANT --------
  HF model:   Qwen/Qwen2.5-7B
  Data:       OpenThoughts + MATH train split
  Algorithm:  COLD-RL with (i,k) text prefix conditioning

  PREFIX FORMAT:
    "[PARALLEL SAMPLE {i} OF {k}]"
    Injected as text tokens at position 0 of input, before problem.
    Examples:
      k=1, i=1:  "[PARALLEL SAMPLE 1 OF 1]" -> max quality, no diversity
      k=8, i=3:  "[PARALLEL SAMPLE 3 OF 8]" -> third of 8 parallel attempts
    The model learns what role i-of-k means purely from reward signal.
    NO explicit mode labels. NO teacher distillation. NO predefined modes.
    This is the key distinction from ModC.

  REWARD:
    R(y_i, S_k) = r_q(y_i) + lambda(k) * r_d(y_i, S_k \ {y_i})
    r_q: binary correctness (1/0) from verifier
    r_d: mean pairwise cosine distance from E5-large-instruct embeddings
         of the OTHER k-1 samples in the group (stop-gradient on encoder)
    lambda(k) = 0.3 * log(k) / log(16)
    lambda(1) = 0 exactly -> k=1 training = standard RL, no diversity pressure

  TRAINING SCHEDULE:
    Phase 1 (steps 1-4,000):    k=1, i=1 only.  Pure quality warmup.
    Phase 2 (steps 4,001-8,000): k ~ Uniform{1,2,4,8,16}, i~Uniform{1..k}
    One k per batch. All k samples of a prompt generated simultaneously.

  LoRA:  rank=32, alpha=64, target all attention+MLP projections
  LR:    5e-5 (LoRA params)
  tau:   1.0 during training rollouts, 0.7 at inference eval
  Steps: 8,000 total

  IMPLEMENTATION NOTE: The two-phase curriculum, k-varied batch sampling,
  and lambda(k) schedule require code changes to rl/diversity_grpo.py
  beyond what currently exists. Must be implemented before R03 can run.

-- R04: COLD-RL (i,k) Qwen3-14B-Base (GPU B, 54h) --------------------------
  HF model:   Qwen/Qwen3-14B
  Data:       DeepscaleR
  Algorithm:  Identical to R03 but on 14B
  Optimizer:  8-bit AdamW. Batch: 32 prompts x k rollouts.
  Purpose:    Beat DARLING 14B. This is the headline comparison.
  Expected:   DARLING reports simultaneous pass@1 + pass@128 improvements.
              COLD-RL should match pass@1 AND beat pass@128 by larger margin
              because DARLING's unconditional diversity cannot shut off at k=1.

-- R05: ModC Reproduction Qwen2.5-7B-Base (GPU C, 22h) ----------------------
  HF model:   Qwen/Qwen2.5-7B
  Data:       OpenThoughts (SAME as R01/R03 -- critical for fair comparison)
  Algorithm:  ModC with mode-specific prefixes
              Two modes trained via distillation from two distinct teachers.
              Teacher A: standard chain-of-thought (Qwen2.5-72B-Instruct)
              Teacher B: step-by-step verification style (same model, different prompt)
              Prefix A: "[REASONING MODE: direct]"
              Prefix B: "[REASONING MODE: verify]"
              Train with SFT on teacher-distilled pairs, mode prefix included.
  Steps:      Same as their paper (4 epochs on OpenThoughts)
  Purpose:    Direct replication of their Table 1 numbers on our hardware.
              If our reproduction matches their reported pass@k, comparison is valid.
  NOTE:       ModC uses distillation (SFT), not RL. Separate training script required.

-- R06: DARLING Reproduction Qwen3-4B-Base (GPU B, 24h) ---------------------
  HF model:   Qwen/Qwen3-4B
  Data:       DeepscaleR (exact match to their paper)
  Algorithm:  DARLING (arXiv:2509.02534)
              Semantic equivalence classifier: Qwen3-Embedding-4B
              Fine-tune classifier on DeepscaleR pairs annotated by
              Llama-3.3-70B-Instruct (their exact setup)
              Reward fusion: MULTIPLICATIVE  r_total = r_q x (1 + gamma*r_div)
              gamma: use their reported value from appendix
  Steps:      8,000
  Purpose:    Direct comparison to their Table 1. Same model, same data.
  NOTE:       Must build/train their semantic equivalence classifier FIRST
              (~4h preprocessing before training starts). Schedule this prep
              on 24GB GPU while R01 runs on GPU C.

-- R07: COLD-RL (i,k) Qwen3-4B-Base (GPU B, 24h) ---------------------------
  HF model:   Qwen/Qwen3-4B
  Data:       DeepscaleR
  Algorithm:  Identical to R03 (same prefix format, same reward, same curriculum)
  Purpose:    Beat DARLING's 4B results. DARLING uses Qwen3-4B so this is
              a perfectly controlled comparison: same model, same data,
              different training algorithm.
  Key metric: pass@k for k in {1,2,4,8,16,64,128} on:
              AIME25, OlympiadBench (DARLING's benchmarks)
              MATH500, GPQA (additional benchmarks for breadth)

-- R08: GRPO Qwen3-4B-Base (GPU B, 18h) -------------------------------------
  HF model:   Qwen/Qwen3-4B
  Data:       DeepscaleR
  Algorithm:  Standard GRPO
  Purpose:    4B baseline. Feeds into DARLING track comparison table.
              Also confirms COLD-RL improves over GRPO at 4B scale.

-- R09: Soft-Token Ablation Qwen2.5-7B-Base (GPU C, 24h) --------------------
  Algorithm:  Replace (i,k) text prefix with soft token e(k) = f_psi(log k)
              injected at embedding layer. ALL k samples get SAME embedding.
              No per-sample role index i.
  Purpose:    Proves per-sample role assignment (the i component) matters
              beyond just knowing the group size k.
              Expected: lower pass@k at k>2 because all samples get same
              signal and cannot differentiate their roles.
              Also: soft token cannot generalize to unseen k via text
              understanding -- must extrapolate via MLP, which is weaker.

-- R10: k-Only Prefix Ablation Qwen2.5-7B-Base (GPU C, 24h) -----------------
  Algorithm:  Prefix is "[GROUP SIZE: k]" -- all k samples in a group
              see identical text prefix. No role index i.
              Same reward structure as R03.
  Purpose:    Isolates contribution of role index i vs knowing group size.
              If R03 >> R10 on pass@k: the i signal is doing real work.
              If R03 ~= R10: diversity comes from reward alone, role index
              is redundant -> simplifies the method.
              Either result is publishable; R03 > R10 strengthens the paper.

-- R11: No Phase-1 Curriculum (GPU C, 24h) -----------------------------------
  Algorithm:  R03 but k sampled from {1,2,4,8,16} from step 1.
              No quality warmup phase.
  Purpose:    Quantify pass@1 cost of skipping curriculum.
              Expected: -3 to -5pp pass@1 with similar or lower pass@16.
              Confirms the two-phase design is necessary.

-- R12: Fixed Lambda Ablation (GPU C, 24h) -----------------------------------
  Algorithm:  R03 but lambda=0.3 for all k (no k-dependent scaling).
              Diversity pressure identical at k=1 and k=16.
  Purpose:    This run MIMICS what DARLING does (fixed unconditional diversity).
              Shows: at k=1, fixed lambda degrades quality (diversity pressure
              when only one sample is drawn is harmful). At high k, fixed
              lambda may be comparable to or worse than COLD-RL's scaled lambda.
              Expected: -2 to -4pp pass@1, similar or slightly lower pass@16.

-- R13: Lexical Diversity Reward (GPU C, 24h) --------------------------------
  Algorithm:  R03 but r_d uses unigram/bigram F1 novelty score instead of
              E5-large-instruct cosine distance.
              Specifically: r_d(y_i, S_k\{y_i}) = avg BLEU-4 complement.
  Purpose:    Proves semantic embedding is necessary. Lexical novelty rewards
              different surface forms of the same solution -- which is NOT
              what we want. Expected: lower gains on hard benchmarks (AIME25,
              GPQA) where the model learns to paraphrase rather than
              discover new solution strategies.

-- R14: Puri et al. Multi-Answer RL (GPU C, 24h) ----------------------------
  HF model:   Qwen/Qwen2.5-7B
  Data:       OpenThoughts (same as R01/R03)
  Algorithm:  Multi-Answer RL (arXiv:2603.24844)
              Single forward pass with K=8 structured outputs
              "[Answer 1]: ... [Answer 2]: ... [Answer 8]: ..."
              Set-level correctness reward
  Purpose:    Beat them at pass@k for k != 8. Their K is fixed at training.
              At k=1: COLD-RL R03 should win (our k=1 = pure quality mode).
              At k=8: comparable (their home turf).
              At k=16: COLD-RL R03 should clearly win (they must run 2
              forward passes; our k=16 is natively trained).
  Key metric: Compare pass@k at k=1,2,4,8,16,64.

-- R15: Zero-Shot (i,k) Diagnostic (GPU A, 2h) -- FREE GO/NO-GO -------------
  Model:      Qwen/Qwen2.5-7B (no fine-tuning, raw base model)
  Method:     Inference only. Test 3 arms on 100 MATH500 problems:
              Arm A: "[PARALLEL SAMPLE i OF 8]" for i=1..8, tau=0.7
              Arm B: 8 i.i.d. samples, no prefix, tau=0.7
              Arm C: 8 i.i.d. samples, no prefix, tau=1.1
  Metric:     Mean pairwise semantic distance, pass@8
  Purpose:    GO/NO-GO gate. Runs in 2h on 24GB before any training starts.
              If Arm A >> Arm B on diversity: (i,k) prefix has semantic
              meaning to the base model before any fine-tuning. Proceed.
              If Arm A ~= Arm B: the prefix is ignored by the base model.
              Adjust prefix wording and re-test before launching R03.
              DO NOT start R03 until R15 confirms the prefix works.

-- R16: COLD-RL (i,k) Qwen2.5-Math-7B (GPU B, 26h) -------------------------
  HF model:   Qwen/Qwen2.5-Math-7B
  Data:       MATH training split (matches Power Sampling EXACTLY)
  Algorithm:  Identical to R03
  Purpose:    Power Sampling uses Qwen2.5-Math-7B. This run beats them
              on their exact model.
  NOTE:       Requires short extra GRPO run on Math-7B (~18h on GPU B)
              to generate the Power Sampling baseline. Count as R20 (buffer).

-- R17: Power Sampling (GPU A, 0h training) ----------------------------------
  Training:   NONE -- inference-only applied to R01 checkpoint
  Algorithm:  Power Sampling MCMC reweighting (ICLR 2026 Vsgq2ldr4K)
              alpha=4.0, Tmax=3072, block_size=192 (their reported values)
              proposal_LLM = base model (Qwen2.5-7B-Base)
  Run on:     24GB GPU (inference only, base model in 4-bit)
  Purpose:    Free competitor. Shows training-based COLD-RL beats
              inference-time redistribution on all k.

-- R18: alpha=0.1 Sensitivity (GPU C, 24h) ----------------------------------
  Algorithm:  R03 with alpha=0.1. Lower diversity pressure.
  Purpose:    Sensitivity analysis. Does COLD-RL degrade with lower alpha?
              Expected: pass@16 slightly lower, pass@1 slightly higher.
              If insensitive: robust result. If sensitive: tune carefully.

-- R19: alpha=0.5 Sensitivity (GPU C, 24h) ----------------------------------
  Algorithm:  R03 with alpha=0.5. Higher diversity pressure.
  Purpose:    Upper bound of diversity weight. Expected: pass@1 degrades
              as diversity pressure at k=1 is non-negligible despite lambda(1)=0,
              because Phase 2 training at k>1 bleeds into k=1 behavior.
              Shows alpha=0.3 is near-optimal.

-- R20: BUFFER ---------------------------------------------------------------
  Reserved for:
    (a) GRPO Qwen2.5-Math-7B baseline if R16 needs it (~18h GPU B)
    (b) Rerun of any diverged training
    (c) Additional benchmark eval if reviewers request specific comparison

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: GPU SCHEDULING -- DAY-BY-DAY GANTT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Hours ->  0    24   48   72   96  120  144  168  192  216  240
            |    |    |    |    |    |    |    |    |    |    |

  GPU A     [R15 ]
  (24GB)    [---- eval R01 ----][eval R02][eval R03][eval R04][eval R05 ->]
            <- runs continuously evaluating each checkpoint as it saves ->
            [R17: Power Sampling inference, ~8h, scheduled after R01 done]

  GPU B     [-------- R02: GRPO 14B --------][------------ R04: COLD-RL 14B ------]
  (96GB)    ..after R04: [R08:GRPO 4B][R06:DARLING][R07:COLD-RL 4B][R16:Math-7B]
            ..if buffer:  [R20: GRPO Math-7B base for R16]

  GPU C     [- R01:GRPO 7B -][------ R03:COLD-RL MAIN * ------][R05:ModC][R09]
  (80GB)    ..[R10:k-only][R11:NoCurr][R12:FixedLambda][R13:Lexical][R14:Puri][R18][R19]

  DECISION POINTS:

  Hour 0:   All three GPUs start simultaneously.
            GPU A: R15 zero-shot diagnostic          [0h -> 2h]
            GPU B: R02 GRPO Qwen3-14B-Base           [0h -> 40h]
            GPU C: R01 GRPO Qwen2.5-7B-Base          [0h -> 20h]

  Hour 2:   R15 completes. DECISION POINT.
            IF GO  -> GPU A begins continuous eval of R01/R02 checkpoints
            IF NO-GO -> revise prefix wording, rerun R15 (costs 2h max)

  Hour 20:  R01 completes on GPU C.
            GPU C: immediately start R03 (COLD-RL main)   [20h -> 46h]
            GPU A: apply Power Sampling (R17) to R01      [20h -> 28h]

  Hour 40:  R02 completes on GPU B.
            GPU B: immediately start R04 (COLD-RL 14B)    [40h -> 94h]
            MILESTONE: check R01 vs R03 intermediate checkpoints.

  Hour 46:  R03 completes on GPU C. * FIRST KEY RESULT
            GPU C: immediately start R05 (ModC repro)     [46h -> 68h]
            GPU A: full eval R03 -- all benchmarks, all k
            DECISION: if R03 >> R01 at ALL k -> proceed with ablations
                      if R03 beats high k but not pass@1 -> adjust Phase 1 ratio

  Hour 68:  R05 (ModC) completes.
            GPU C: start R09 (soft-token ablation)        [68h -> 92h]
            MILESTONE: R03 vs R05 ModC comparison complete.

  Hour 94:  R04 (COLD-RL 14B) completes on GPU B. * SECOND KEY RESULT
            GPU B: start R08 (GRPO Qwen3-4B baseline)    [94h -> 112h]
            MILESTONE: COLD-RL 14B vs DARLING comparison ready.

  Hours 92-240: Ablations + secondary competitors, sequential on GPU C.
            GPU C queue: R09 -> R10 -> R11 -> R12 -> R13 -> R14 -> R18 -> R19
            GPU B queue: R06 -> R07 -> R16 -> R20 (if needed)

  TOTAL WALL-CLOCK:
    Critical path (GPU B): ~160h = 6.7 days to have all main results
    Full ablation suite:   ~240h = 10 days to complete all 20 runs
    Paper-writing can start at hour 94 when R03 + R04 + R05 are done.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5: BENCHMARKS AND WHICH PAPER EACH COVERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  BENCHMARK       COVERS                               EVAL GPU
  ----------------------------------------------------------------
  MATH-500        Power Sampling, ModC, baseline        A (24GB)
  AIME25          DARLING (their primary benchmark)     A (24GB)
  OlympiadBench   DARLING (their primary benchmark)     A (24GB)
  GPQA-Diamond    Power Sampling, generalization        A (24GB)
  HumanEval+      Power Sampling, ModC (code)           A (24GB)
  NuminaMath      ModC (their secondary benchmark)      A (24GB)

  All benchmarks evaluated from every checkpoint.
  Estimator: unbiased Chen et al. (2021) with n=200 per problem.
  k values:  {1, 2, 4, 8, 16, 32, 64, 128} -- include 32, 64, 128 for
             generalization to unseen k (trained on k<=16).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6: THE COMPARISON TABLES EACH RUN FEEDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TABLE 1: Main results -- Qwen2.5-7B (beats ModC + Power Sampling)
    Rows:    GRPO(R01), ModC(R05), Power Sampling(R17), COLD-RL(R03)
    Cols:    pass@k for k={1,2,4,8,16,64} on MATH500, GPQA, HumanEval+
    Claim:   COLD-RL Pareto-dominates all rows at every column.

  TABLE 2: DARLING comparison -- Qwen3-4B (beats DARLING 4B)
    Rows:    GRPO(R08), DARLING(R06), COLD-RL(R07)
    Cols:    pass@k for k={1,2,4,8,16,64,128} on AIME25, OlympiadBench
    Claim:   COLD-RL beats DARLING at every k, including pass@1 AND pass@128.

  TABLE 3: Scale -- Qwen3-14B (beats DARLING 14B)
    Rows:    GRPO(R02), DARLING (cited from paper), COLD-RL(R04)
    Cols:    pass@k for k={1,2,4,8,16,64,128} on AIME25, OlympiadBench

  TABLE 4: Ablation study -- Qwen2.5-7B (MATH500 pass@1 / pass@16)
    Rows:    COLD-RL(i,k)(R03) | k-only pfx(R10) | soft-token(R09) |
             no Phase-1(R11) | fixed lambda(R12) | lexical r_d(R13)
    Purpose: Every ablation should show R03 wins, validating each component.

  TABLE 5: Competitor -- Multi-Answer RL (Puri et al.) vs COLD-RL
    Rows:    Puri et al.(R14), COLD-RL(R03)
    Cols:    pass@k for k={1,2,4,8,16,64} -- show COLD-RL wins at k != 8

  TABLE 6: Generalization to unseen k (k=32, 64, 128)
    Inference only from R03 checkpoint. No new runs needed.
    Proves f(i,k) prefix generalizes via ordinal text understanding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7: EXACT LAUNCH COMMANDS (CURRENT CODEBASE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NOTE: Current codebase uses rl/run_rl.py (GRPO + fixed lambda diversity).
  R03's two-phase curriculum and k-varied training require code changes to
  rl/diversity_grpo.py. R05 and R14 require separate SFT training scripts.

  R01 on GPU C -- GRPO baseline (Qwen2.5-7B base):
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup python3 -u -m rl.run_rl \
      --model Qwen/Qwen2.5-7B \
      --dataset deepscaler \
      --n_rollouts 8 --temperature 1.0 \
      --total_steps 8000 --n_problems_per_step 64 \
      --mini_batch 16 --ref_batch_size 4 \
      --max_new_tokens 4096 \
      --lr 1e-6 --kl_beta 0.04 \
      --embed_model local --lora_r 32 --lora_alpha 64 \
      --save_every 2000 --lambda_div 0.0 \
      --output_dir results/rl_runs/r01_grpo_7b \
      --gpu_id 0 > logs/r01_grpo_7b.log 2>&1 &

  R03 on GPU C -- COLD-RL (once two-phase curriculum is implemented):
    Identical flags to R01 except:
      --lambda_div 0.5 (will be overridden by lambda(k) schedule in code)
      --output_dir results/rl_runs/r03_cold_rl_7b
      > logs/r03_cold_rl_7b.log

  R15 on GPU A -- Zero-shot diagnostic (run this first on the 24GB GPU):
    python3 scripts/r15_zero_shot_diagnostic.py \
      --model Qwen/Qwen2.5-7B \
      --n_problems 100 \
      --dataset math500 \
      --n_samples 8 \
      --temperature 0.7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 8: RISKS AND MITIGATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RISK 1: R03 improves pass@k but not pass@1 vs GRPO
  MITIGATION: Increase Phase 1 steps from 4000 to 5000 (use R20 buffer).
              The Phase 1 / Phase 2 split is the primary knob for pass@1.

  RISK 2: (i,k) prefix ignored by base model (R15 NO-GO)
  MITIGATION: Add explicit instruction to prefix:
              "[PARALLEL SAMPLE {i} OF {k} -- use a different reasoning
               strategy than earlier samples would use]"
              Rerun R15 (2h on GPU A). Then proceed.

  RISK 3: DARLING reproduction doesn't match their reported numbers
  MITIGATION: Flag as "independently reproduced" with noted deviation.
              COLD-RL comparison against our reproduced baseline is still valid.

  RISK 4: Qwen3-14B doesn't fit on 96GB even with 8-bit optimizer
  MITIGATION: Use LoRA rank=16 instead of 32. Or QLoRA (4-bit base).
              Last resort: gradient_accumulation_steps=16, micro_batch=2.

  RISK 5: COLD-RL loses to ModC at pass@1
  MITIGATION: (a) Extend Phase 1 to 5000 steps. (b) Lower alpha to 0.1.
              If still losing: report honestly. pass@1 >= GRPO is sufficient;
              the main claim is pass@k for k>1 via controllable diversity.
