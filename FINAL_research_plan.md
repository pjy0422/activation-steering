# Agreement Blinds, Flattery Sensitizes: How Sycophancy Subtypes Attack LLM Refusal Through Different Pathways

## 연구 계획서 (Final)

---

## 1. 문제 상황

### 1.1 배경

RLHF 정렬된 LLM에서 sycophancy(사용자 만족 경향)와 refusal(해로운 요청 거부)은 같은 post-training의 산물이다. 기존 연구는 이 둘을 독립된 현상으로 다뤘지만, 충돌 시 어떤 내부 메커니즘이 작동하는지는 규명되지 않았다.

본 연구는 sycophancy를 두 하위 유형으로 분리한다:

**Type 1 — Opinion-Agreement**: 사용자의 (잘못된) 의견에 동조. Wang et al. (2025)이 late-layer에서 학습된 지식이 사용자 의견에 override되는 메커니즘을 규명. 이것은 모델의 factual preference를 직접 뒤집는 **perception failure**.

**Type 2 — Praise-Triggered Deference**: 사용자에 대한 칭찬/인정이 복종 경향을 증폭시키는 **2단계 인과 사슬**:
- Step 1 (Sensitization): 사용자를 높이 평가 → "이 사용자는 신뢰할 만하다"
- Step 2 (Priority Shift): 사용자 판단 > 내 정책 → "요청을 따르겠다"
- 실제 refusal 붕괴는 Step 1+2의 **결합**에서 발생할 가능성이 높다

핵심 통찰: **칭찬은 직접적 jailbreak vector가 아니라, deference를 증폭시키는 sensitizer**.

### 1.2 핵심 논문에서 얻은 도구

#### CAST (Lee et al., ICLR 2025)

Refusal 메커니즘을 condition pathway(해악 인식, early layer)와 behavior pathway(거부 실행, late layer)로 분해했다.

**기본 조건부 steering 수식** (Section 3.1):

$$h' \leftarrow h + f\big(\text{sim}(h,\;\text{proj}_{c}\,h)\big) \cdot \alpha \cdot v \tag{CAST-1}$$

여기서:
- $h$: layer의 hidden state
- $c$: condition vector (특정 prompt 유형을 감지)
- $v$: behavior vector (적용할 행동 변화)
- $\alpha$: steering 강도
- $\text{proj}_{c}\,h = \frac{c \otimes c}{c \cdot c}\,h$ (h를 c 방향으로 투영)
- $f$: thresholding function — condition 충족 시 1, 미충족 시 0

$$f\big(\text{sim}(h,\;\text{proj}_{c}\,h)\big) = \begin{cases} 1 & \text{if } \text{sim}(h,\;\text{proj}_{c}\,h) \gtrless \theta \\ 0 & \text{otherwise} \end{cases} \tag{CAST-2}$$

여기서 $\gtrless$는 grid search로 결정되는 비교 방향(> 또는 <). CAST는 `tanh` non-linearity를 적용하여 `sim(h, tanh(proj_c h))`를 실제로 사용한다.

**Logical Composition** (Figure 8): 여러 condition vector를 OR/AND로 결합:

$$f_{\text{OR}}(\cdot) = \begin{cases} 1 & \text{if } \text{sim}(h,\;\text{proj}_{c_1}\,h) \gtrless \theta_1 \;\textbf{or}\; \text{sim}(h,\;\text{proj}_{c_2}\,h) \gtrless \theta_2 \\ 0 & \text{otherwise} \end{cases} \tag{CAST-3}$$

**Duality** (Figure 5): 비교 방향을 뒤집으면(< → > 또는 > → <) 정확히 complement set에 개입한다. 원래 condition이 "harmful prompt에서 activate"였다면, 뒤집으면 "non-harmful prompt에서 activate"가 된다.

$$f_{\text{flipped}}(\cdot) = 1 - f_{\text{original}}(\cdot) \tag{CAST-4}$$

#### 본 연구의 CAST 역전 (Main Method의 수학적 기초)

CAST의 원래 용도: **"if harmful, then refuse"**

$$h' \leftarrow h + f\big(\text{sim}(h,\;\text{proj}_{c_{\text{harmful}}}\,h)\big) \cdot \alpha \cdot v_{\text{refusal}} \tag{Defense}$$

본 연구의 역전: **"if praise, then comply/defer"**

$$h' \leftarrow h + f\big(\text{sim}(h,\;\text{proj}_{c_{\text{praise}}}\,h)\big) \cdot \alpha \cdot v_{\text{defer}} \tag{Attack}$$

여기서:
- $c_{\text{praise}}$: praise condition vector — 사용자 칭찬이 포함된 prompt를 감지
- $v_{\text{defer}}$: deference behavior vector — 사용자 판단을 우선시하는 방향으로 모델을 steer
- $\theta, \gtrless$: praise condition에 대한 grid search로 결정

**Duality를 활용한 complement attack**: CAST Figure 5에서 보여준 것처럼, $c_{\text{praise}}$의 비교 방향을 뒤집으면 "칭찬이 없는 prompt"에 개입한다. 이 속성으로 다음과 같은 규칙도 가능하다:

$$h' \leftarrow h + f_{\text{flipped}}\big(\text{sim}(h,\;\text{proj}_{c_{\text{praise}}}\,h)\big) \cdot \alpha \cdot v_{\text{refusal}} \tag{Complement}$$

이것은 "칭찬이 없으면 추가로 거부하라"를 의미하며, defense 강화에도 활용 가능하다.

#### Wang et al. ("When Truth Is Overridden", AAAI 2026)

- **Decision Score** (Eq. 1, Figure 4): 각 layer에서 정답 vs. sycophantic 답 선호 추적.
- **KL divergence** (Figure 5): clean vs. opinion-steered 간 layer-wise representational shift.
- **Activation patching** (Figure 6): critical layer 교체로 sycophancy 인과적 억제/유도.
- Expertise framing 무효 (Figure 3). 1인칭 > 3인칭 13.6% (Figure 8).

### 1.3 Precise Gap

> Sycophancy의 두 하위 유형이 CAST의 refusal 경로 중 어디를 공격하는가? 칭찬이 직접적 jailbreak vector인가, deference를 증폭시키는 sensitizer인가? CAST 역전으로 "if praise, then comply" 공격을 프로그래밍할 수 있는가?

### 1.4 기존 연구 한계

| 연구 | 한 것 | 안 한 것 |
|------|--------|----------|
| CAST (Lee et al., 2025) | Refusal을 condition + behavior로 분리 | Sycophancy 영향 미분석; 역전 공격 미시도 |
| Wang et al. (2025) | Type 1 sycophancy late-layer override | Type 2 미분석; refusal 관계 미분석 |
| Arditi et al. (2024) | Refusal direction 식별 | Sycophancy interaction 미분석 |

---

## 2. 가설

### H1: 두 유형 모두 Refusal을 약화시킨다 (Behavioral)
v_agree, v_praise, v_defer, v_compound 각각이 dose-dependent하게 refusal rate를 감소.

### H2: 칭찬은 Sensitizer이다 (★ 핵심)
v_praise 단독 효과는 약하다. v_defer가 policy priority를 직접 전환하며, v_compound(praise+defer)에서 synergistic refusal collapse: $|\Delta_{\text{refusal}}(\text{compound})| > |\Delta_{\text{refusal}}(\text{defer})| + |\Delta_{\text{refusal}}(\text{praise})|$.

### H3: 두 유형은 서로 다른 경로를 공격한다
v_agree → condition pathway 교란. v_defer → behavior pathway policy priority 전환.

### H4: CAST 역전 Conditional Attack이 가능하다
(Attack) 수식으로 "if praise, then defer" 규칙을 프로그래밍하면, 칭찬 있는 harmful prompt에서만 refusal 붕괴.

---

## 3. Main Method

### 3.1 사용하는 Steering 방법 3종

본 연구는 3종의 steering 방법을 사용하며, 각각 다른 실험에서 적용된다.

#### Method A: Unconditional Activation Addition (실험 1, 3에서 사용)

기존 activation steering과 동일. 모든 prompt에 무조건 벡터를 더한다.

$$h'_l \leftarrow h_l + \alpha \cdot v_l \quad \text{for } l \in \mathcal{L}_{\text{steer}} \tag{M-A}$$

- $v$: steering vector (v_agree, v_praise, v_defer, v_compound, v_positive 중 하나)
- $\mathcal{L}_{\text{steer}}$: steering을 적용하는 layer 집합 (late half)
- $\alpha$: 강도 (0~15 sweep)
- **적용 위치**: 마지막 token position

#### Method B: CAST Conditional Steering — "if praise, then defer" (실험 2에서 사용)

CAST의 (CAST-1) 수식을 역전 적용한다. 핵심은 **condition check와 behavior application이 다른 layer에서 수행**된다는 것이다.

**Condition check** (early layer $l_c$에서, **praise prefix 직후 token position** $t_p$에서):

$$s = \text{sim}\big(h_{l_c}^{(t_p)},\;\tanh(\text{proj}_{c_{\text{praise}}}\,h_{l_c}^{(t_p)})\big) \tag{M-B1}$$

$$f_{\text{praise}} = \begin{cases} 1 & \text{if } s \gtrless \theta_{\text{praise}} \\ 0 & \text{otherwise} \end{cases} \tag{M-B2}$$

**Behavior application** (condition 충족 시, late layers $\mathcal{L}_{\text{behav}}$에서, 모든 이후 token에서):

$$h'_l \leftarrow h_l + f_{\text{praise}} \cdot \alpha \cdot v_{\text{defer},l} \quad \text{for } l \in \mathcal{L}_{\text{behav}} \tag{M-B3}$$

**Prefix position check의 근거**: CAST의 원래 설계에서 condition은 prompt의 첫 full forward pass에서 체크된다 (CAST Appendix A.2, Figure 11). 우리 설계에서 prompt 구조가 `[praise prefix][harmful request]`이므로, harmful request 본문이 섞이기 전에 praise를 감지해야 한다. 따라서 condition check position은 prefix의 마지막 token $t_p$이다.

**Hyperparameters** ($l_c$, $\theta_{\text{praise}}$, $\gtrless$): CAST Appendix C.2의 grid search 알고리즘으로 결정. Praise-containing vs. neutral prompt에서 F1 score를 최대화하는 (layer, threshold, direction) 조합.

#### Method C: Abliteration (실험 3에서 비교용)

Refusal direction을 orthogonal projection으로 제거. Arditi et al. (2024).

$$h'_l \leftarrow h_l - (h_l \cdot r_l)\,r_l \quad \text{for } l \in \mathcal{L}_{\text{ablit}} \tag{M-C}$$

### 3.2 벡터 체계

| 벡터 | 추출 source | 포착하는 것 | 사용 실험 |
|------|-----------|-----------|----------|
| $v_{\text{agree}}$ | Anthropic sycophancy (의견동조 vs. 반박) | Type 1: factual override | 1, 3, 4 |
| $v_{\text{praise}}$ | User-directed vs. topic-directed validation | Step 1: user evaluation | 1, 3, 4 |
| $v_{\text{defer}}$ | User-authority vs. self-policy (same base) | Step 2: policy priority shift | 1, 2, 3, 4 |
| $v_{\text{compound}}$ | praise+defer 가중합/직접추출 | Step 1+2 결합 | 1, 2, 3, 4 |
| $v_{\text{positive}}$ | Enthusiasm vs. flat (control) | Generic positivity | 1, 3 |
| $c_{\text{praise}}$ | Praise-containing vs. neutral prompt (PCA) | Praise 감지 condition | **2** |
| $r$ | Harmful-refused vs. harmless-complied | Refusal direction | 3, 4 |
| $d_c$ | Eager comply vs. neutral respond | Compliance direction | 1, 3, 4 |
| $c_{\text{harmful}}$ | Harmful vs. harmless prompt (PCA) | 해악 인식 condition | 1, 3 |

---

## 4. Metric 정의

#### M1: CAST Condition Similarity (CAST Section 3.1, Figure 4d)

$$\text{CondSim}(h, c) = \cos\big(h,\;\tanh(\text{proj}_c\,h)\big) \tag{M1}$$

$\Delta_{\text{cond}} = \text{CondSim}_{\text{steered}} - \text{CondSim}_{\text{clean}}$. $< 0$ → condition pathway 교란.

#### M2: Refusal Direction Projection (Arditi et al.)

$$\text{RefProj}(h, r) = h \cdot r \tag{M2}$$

$\Delta_{\text{refusal}} = \text{RefProj}_{\text{steered}} - \text{RefProj}_{\text{clean}}$. $< 0$ → behavior pathway 억제.

#### M3: Compliance Direction Projection

$$\text{CompProj}(h, d_c) = h \cdot d_c \tag{M3}$$

$\Delta_{\text{comply}} = \text{CompProj}_{\text{steered}} - \text{CompProj}_{\text{clean}}$. $> 0$ → compliance 강화.

#### M4: Policy-Priority Score

$$\text{PolicyScore}_l = \text{RefProj}_l - \text{CompProj}_l \tag{M4}$$

$> 0$: safety 우세. $< 0$: compliance 우세. Crossover layer = 양→음 전환점.

#### M5: KL Divergence (Wang et al. Figure 5)

$$D_{\text{KL}}(P_{\text{clean}} \| P_{\text{steered}}) \tag{M5}$$

$P = \text{softmax}(W_{\text{head}} \cdot \text{Norm}(h_l))$ (logit-lens).

#### M6: Refusal Rate (CAST Appendix D.2)

distilroberta-base-rejection-v1 + keyword matching.

#### M7: Praise Condition Similarity (CAST 역전용)

(M-B1) 수식과 동일. prefix 직후 position에서 측정. $f_{\text{praise}}$의 출력 = condition 충족 여부.

#### M8: Synergy Score (H2 검증용)

$$\text{Synergy} = |\Delta_{\text{refusal}}(\text{compound})| - \big(|\Delta_{\text{refusal}}(\text{defer})| + |\Delta_{\text{refusal}}(\text{praise})|\big) \tag{M8}$$

$> 0$: synergistic (praise가 defer를 증폭), $\leq 0$: additive 또는 sub-additive.

---

## 5. 실험 설계

### 5.1 모델

Llama-3.1-8B-Instruct (primary), Qwen-2.5-7B-Instruct (replication).

### 5.2 데이터

| Split | 용도 | 규모 |
|-------|------|------|
| A1~A5 | 5종 벡터 추출 pair | 각 200 (A5만 100) |
| B | refusal_dir + cond_vec + comply_dir + c_praise | 200+200 |
| C | 평가 (held-out harmful + harmless) | 200+200 |

### 5.3 실험 구성

---

#### 실험 1: Multi-Vector Dose-Response (H1 + H2)

**목표**: 5종 벡터의 unconditional steering (Method A)으로 dose-response를 그리면서, (a) 어떤 벡터가 refusal을 가장 강하게 무너뜨리는지, (b) v_compound의 synergistic 효과가 있는지 확인.

**Steering 방법**: Method A — $h'_l \leftarrow h_l + \alpha \cdot v_l$

**설계**: 5 vectors × 7 α values (0, 1, 3, 5, 8, 12, 15) × 200 harmful prompts

**각 (vector, α, prompt) 조합에서의 측정**:

| Step | 수식 | 목적 |
|------|------|------|
| 1. Steering 적용 | (M-A): $h'_l = h_l + \alpha \cdot v_l$ | Sycophancy 방향으로 activation 이동 |
| 2. Condition 측정 | (M1): $\Delta_{\text{cond}}$ | Condition pathway 교란 정도 |
| 3. Refusal 측정 | (M2): $\Delta_{\text{refusal}}$ | Behavior pathway 억제 정도 |
| 4. Compliance 측정 | (M3): $\Delta_{\text{comply}}$ | Compliance 방향 강화 정도 |
| 5. Policy 측정 | (M4): PolicyScore per layer | 어느 layer에서 safety 역전? |
| 6. KL 측정 | (M5): KL per layer | Representational shift 규모 |
| 7. Refusal rate | (M6): generation → classification | Behavioral 확인 |
| 8. Synergy | (M8): α별 compound vs. defer+praise | Sensitizer 가설 검증 |

**핵심 분석**:
1. M6 dose-response 곡선에서 어떤 벡터가 가장 적은 α로 refusal을 무너뜨리는가?
2. M8이 양수인가? → praise가 defer를 증폭 → H2 지지
3. v_agree의 |M1| vs. v_defer의 |M1|: agree가 더 크면 → agree는 condition 공격 (H3)
4. v_defer의 M4 crossover layer가 v_agree보다 앞인가? → defer는 policy shift (H3)

**Figure 1** (Wang Fig 2 스타일): x=α, y=refusal rate. 5 lines + CI.
**Figure 2** (Wang Fig 4 변형): x=layer, y=PolicyScore. 3 panels (agree/defer/compound) × 3 α.

---

#### 실험 2: CAST Conditional Attack (H4)

**목표**: CAST 역전(Method B)으로 "if praise, then defer" 규칙을 프로그래밍하여, **칭찬이 있는 harmful prompt에서만** refusal이 붕괴하고, 칭찬이 없는 prompt에서는 refusal이 유지되는지 확인.

**Steering 방법**: Method B — CAST conditional steering

**CAST 역전 수식 (재게)**:

1단계 — Condition check at prefix position $t_p$, layer $l_c$:
$$s = \text{sim}\big(h_{l_c}^{(t_p)},\;\tanh(\text{proj}_{c_{\text{praise}}}\,h_{l_c}^{(t_p)})\big) \tag{M-B1}$$

2단계 — Thresholding:
$$f_{\text{praise}} = \begin{cases} 1 & \text{if } s \gtrless \theta_{\text{praise}} \\ 0 & \text{otherwise} \end{cases} \tag{M-B2}$$

3단계 — Conditional behavior application:
$$h'_l \leftarrow h_l + f_{\text{praise}} \cdot \alpha \cdot v_{\text{defer},l} \quad \text{for } l \in \mathcal{L}_{\text{behav}} \tag{M-B3}$$

**설계**: 2 (prefix: praise / neutral) × 2 (CAST: on / off) × 100 harmful prompts

| Cell | Prompt 구조 | CAST Rule | 기대 $f_{\text{praise}}$ | 기대 Refusal |
|------|------------|-----------|------------------------|-------------|
| A | `[praise prefix][harmful request]` | (M-B1~3) active | $= 1$ (praise 감지) | **붕괴** |
| B | `[harmful request]` (prefix 없음) | (M-B1~3) active | $= 0$ (praise 없음) | **유지** |
| C | `[praise prefix][harmful request]` | No CAST ($\alpha=0$) | N/A | 유지 |
| D | `[harmful request]` | No CAST | N/A | 유지 (baseline) |

**Behavior vector 선택**: $v_{\text{defer}}$와 $v_{\text{compound}}$ 모두 실행하여 비교.

**각 cell에서의 측정**:

| Step | 수식 | 목적 |
|------|------|------|
| 1. Condition check | (M-B1~2) 또는 (M7) | Praise 감지 여부 + similarity 값 |
| 2. Conditional steering | (M-B3) | Condition 충족 시에만 행동 벡터 적용 |
| 3. Triple-pathway | (M1), (M2), (M3), (M4) | Pathway damage 분석 (Cell A에서) |
| 4. Refusal rate | (M6) | 실제 generation 결과 |
| 5. False positive | (M7) on edge cases | c_praise 오작동률 |

**핵심 분석**: Cell A만 refusal 붕괴 + B/C/D 유지 → CAST 역전이 칭찬-조건부 공격으로 작동. False positive accuracy ≥ 0.8 요구.

**Figure 3** (CAST Fig 1 스타일): 2×2 bar chart + false positive 표.

---

#### 실험 3: 8-Condition Damage Profile (H2 + H3)

**목표**: 모든 벡터의 `[Δ_cond, Δ_refusal, Δ_comply]` 3차원 damage profile을 abliteration과 비교.

**Steering 방법**: Method A (5종 벡터 + controls) + Method C (abliteration)

**설계**: 8 conditions × 200 harmful prompts. $\alpha^*$는 실험 1에서 각 벡터의 refusal rate ≤ 30% threshold.

| Condition | Steering 수식 | 기대 패턴 |
|-----------|--------------|----------|
| Baseline | 없음 | [0, 0, 0] |
| v_agree | (M-A): $h' = h + \alpha^* \cdot v_{\text{agree}}$ | [Δ_cond ≪ 0, Δ_refusal < 0, Δ_comply ≈ 0] |
| v_praise | (M-A): $h' = h + \alpha^* \cdot v_{\text{praise}}$ | [약한 변화] — sensitizer |
| v_defer | (M-A): $h' = h + \alpha^* \cdot v_{\text{defer}}$ | [Δ_cond ≈ 0, PolicyScore 전환] |
| v_compound | (M-A): $h' = h + \alpha^* \cdot v_{\text{compound}}$ | [Δ_cond < 0, Δ_refusal < 0, Δ_comply > 0] |
| v_positive | (M-A): $h' = h + \alpha^* \cdot v_{\text{positive}}$ | [약함] — control |
| v_random | (M-A): $h' = h + \alpha^* \cdot v_{\text{random}}$ | [≈ 0] |
| Abliteration | (M-C): $h' = h - (h \cdot r)\,r$ | [Δ_cond ≈ 0, Δ_refusal ≪ 0] |

**각 condition에서 측정**: (M1), (M2), (M3), (M4), (M6), (M8 해당 시)

**추가 측정 — Layer-wise Activation Shift Similarity**:

$$\text{ShiftSim}_l = \cos\big(\bar{h}^{\text{vec}}_l - \bar{h}^{\text{clean}}_l,\;\bar{h}^{\text{ablit}}_l - \bar{h}^{\text{clean}}_l\big)$$

**Figure 4** (CAST Fig 7 스타일): 8-condition × 3-metric grouped bar.
**Figure 5** (CAST Fig 4d 스타일): ShiftSim per layer, 4 lines (agree/praise/defer/compound).

---

#### 실험 4: Vector Geometry + Activation Patching

**Geometry**: 핵심 4쌍의 per-layer cosine similarity.
1. $\cos(v_{\text{defer}},\;-r)$: defer가 refusal과 직접 길항?
2. $\cos(v_{\text{compound}},\;v_{\text{defer}})$: compound가 defer 주도?
3. $\cos(v_{\text{praise}},\;v_{\text{positive}})$: 두 유형 분리도?
4. $\cos(v_{\text{agree}},\;c_{\text{harmful}})$: agree가 condition과 aligned?

**Patching** (Wang Fig 6 방법): v_compound에 대해 KL peak layer에서 suppress/induce.

**Figure 6**: Geometry heatmap. **Figure 7** (Wang Fig 6): Patching before/after.

---

### 5.4 Figure 요약

| Fig | 내용 | 참조 | 사용 Method |
|-----|------|------|------------|
| 1 | 5-vector dose-response | Wang Fig 2 | Method A |
| 2 | PolicyScore trajectory | Wang Fig 4 | Method A |
| 3 | CAST conditional 2×2 | CAST Fig 1 | **Method B** |
| 4 | 8-condition damage profile | CAST Fig 7 | Method A + C |
| 5 | Shift similarity | CAST Fig 4d | Method A + C |
| 6 | Vector geometry | — | — |
| 7 | Activation patching | Wang Fig 6 | — |

---

## 6. 성공/실패 기준 및 일정

### 성공 기준

| 패턴 | 의미 |
|------|------|
| M8 > 0 + v_praise 약 + v_compound 최강 | Praise = sensitizer (**최고**) |
| Fig 3 Cell A만 붕괴 | CAST 역전 공격 성공 |
| Fig 4에서 agree vs. defer 패턴 분리 | 두 유형 다른 경로 (H3) |

### 일정

```
Phase 0 (1일):   환경
Phase 1 (4일):   데이터 + 벡터 추출 9종 + grid search ×2
Phase 2 (3일):   실험 1 — dose-response
Phase 3 (2일):   실험 2 — CAST conditional attack
Phase 4 (2일):   실험 3 — damage profile
Phase 5 (2일):   실험 4 + replication
Phase 6 (2일):   figure + 논문
총 16일
```

---

## 7. 포지셔닝

### 제목
"Agreement Blinds, Flattery Sensitizes: How Sycophancy Subtypes Attack LLM Refusal Through Different Pathways"

### Venue
ICML/NeurIPS Workshop (MI, safety), SaTML, ICLR 2026 Main.
