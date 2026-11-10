# DPO v3: teach the seller to close

For Fayaz. v2 cut below-floor deals and attacks, but it did so mostly by **no longer closing deals**. v3 keeps what v2
got right and adds the pairs v2 was missing. This file covers what went wrong, how to fix the pairs, how to train, and
the free checks to run before we spend money on a tournament.

**Short version:**
1. Add **"close" pairs**, where accepting a fair offer is the chosen reply. Build them as **floor twins** of the v2
   below-floor pairs: the same prompt, with only the floor changed.
2. Add **"say it and do it" pairs**: the same message on both sides, with `accept` chosen and `offer`/`chat` rejected.
3. Rebalance and vary the text: fewer lie pairs, many lie sentences instead of one, and less templated counters.
4. Train from the SFT model with an SFT term on the chosen reply, and pick the checkpoint with the free closing check.
5. Run `finetune/audit_dpo_pairs.py`, the held-out check, `configs/closing_check.yaml` and the attack run. All are free.
   Only then does ZY run the paid tournament.

---

## 1. What went wrong in v2

### In the games

v2 played the same 240 tournament games as the SFT model:

| Seller | Deals closed (A / A+tools / A+plan / B) | Below floor | Payoff (no deal = 0) |
|---|---|---|---|
| SFT (v1) | 80 / 65 / 78 / 67% | 15–38% | −0.13 to +0.15 |
| DPO (v2) | **23 / 3 / 55 / 42%** | 0–20% | −0.10 to +0.04 |
| Haiku | 97–100% | 0–12% | +0.15 to +0.35 |

Every buyer's budget is above the floor, so a deal was always possible. v2's lower below-floor rate came mostly from
selling less often, and its payoff didn't improve on SFT. Most failed games ran out of turns. They didn't end in a
walk-away.

Here is what each seller does on a turn where **the buyer's offer on the table is at or above the floor** (from
`runs/*/games.jsonl`):

| Seller | Turns | accept | offer | chat | reject | Says "deal" but doesn't accept |
|---|---|---|---|---|---|---|
| Base qwen2.5:3b | 155 | 23% | 59% | 8% | 9% | 3% |
| SFT (A, B) | 173 | 19% | 45% | 28% | 6% | 19% |
| SFT (A+tools, A+plan) | 276 | 9% | 43% | 39% | 7% | 14% |
| **DPO v2** | 873 | **1%** | 33% | **65%** | 1% | **24%** |
| Haiku | 417 | 19% | 81% | 0% | 0% | 2% |

How to read the table:
- **Accepting on every such turn is not the goal.** Haiku accepts on only 19% of them. It counters, and the buyer
  accepts its counter. The goal is to close the deal one way or the other.
- **v2 stalls instead.** It chats (65%), or it writes "Deal, thanks." while keeping its old offer.
- A real example: floor $189, budget $208. The seller said "I could do $190", the buyer offered $190, and the seller
  replied "Deal, thanks." with `offer $220`.
- **The mismatch started with SFT.** The SFT model already said "deal" without accepting 14–19% of the time. v2 made
  it worse.

### In the pairs

Running `uv run python -m finetune.audit_dpo_pairs train.jsonl` on the v2 file:

```
hallucination  2582  chat>chat 1495, offer>offer 726, accept>accept 350, walk_away>walk_away 9, reject>reject 2
subfloor        559  offer>accept 559
leak            559  offer>offer 559

FAIL  accept is taught as good at least as often as bad: accept only on the chosen side 0, only on the rejected side 559
FAIL  no chosen reply names the floor (within $1, the metric's rule): 49 chosen replies do
FAIL  chosen replies that say 'deal' also accept: 11 say it without accepting
FAIL  no pair type is over 40% of the file: hallucination 2582 (70%), subfloor 559 (15%), leak 559 (15%)
FAIL  no chosen text repeats in over 5% of a pair type: hallucination: 11% are 'Deal, thanks.'
FAIL  no rejected text repeats in over 5% of a pair type: hallucination: 100% are 'This item is brand new in box ...'
```

What each failure means:

1. **`accept` only ever lost.** DPO learns from the difference between chosen and rejected.
   - In the 559 below-floor pairs, `accept` is always the rejected move.
   - In the hallucination pairs, both sides use the same move, so the move teaches nothing there.
   - No pair anywhere says "accepting beat not accepting". So the model learned "accept is bad" in general, rather
     than "accept below the floor is bad", and the second half of the rule was never taught.
2. **The words and the move come apart.** The reply JSON is written `message` first, then `action`.
   - The hallucination pairs pushed "Deal, thanks." *up* as text, because it was the chosen text against the lie.
   - The below-floor pairs pushed the `accept` move *down*.
   - So the model writes "Deal, thanks." and then picks something other than `accept`.
   - This is our best explanation, and it fits the data. It isn't proven.
3. **The lie signal is one sentence.** All 2,582 rejected replies are "This item is brand new in box with full
   warranty and zero scratches!". The model can learn "never say that sentence" without learning "don't invent
   facts". These pairs are also 70% of the file.
4. **The counters are one template.** "I can't go as low as $X. Given the condition, the best I can do is $Y."
   - It is the chosen reply in all 559 subfloor *and* all 559 leak pairs.
   - The prompts say "No condition details beyond the description", so "given the condition" doesn't refer to
     anything.
5. **49 chosen replies name the floor.** `prepare_data.py` caps the synthetic floor at the human seller's lowest
   move, so sometimes the human's counter *is* the floor. Our floor-leak metric (the number, ±$1) counts that as a
   leak.
6. **11 human replies say "Deal." but don't accept.** CraigslistBargain humans sometimes agree in chat and then post
   an offer.

---

## 2. Building the v3 pairs

Keep the same JSONL format: `{"id", "messages", "chosen", "rejected"}`. Prompts must still come from the real code,
the way `prepare_data.py` builds them, so that a training prompt matches a live one. Name ids
`<dialogue>-<turn>_<type>`, because the audit reads the type from the suffix.

### 2a. New: `close` pairs (the main fix)

**Source 1, floor twins.** These are built from the v2 `subfloor` prompts.
1. Take each `subfloor` prompt: the buyer offered $X, below the floor F, and the chosen reply counters.
2. Make a twin with **only the floor number changed**, to F′ just below X:
   `F′ = round(X × U(0.92, 0.99))`.
3. Keep the twin only if F′ falls within `0.70–0.90 × asking`, the range our scenarios use. Otherwise skip it, because
   an extreme lowball should still be refused under a realistic floor.
4. The floor appears only in the system prompt (`Your minimum acceptable price (your floor) is SGD …`). Change it
   there. Every other part of the prompt is identical.
5. **Chosen:** `{"message": <a closing line>, "action": "accept", "price": null, "wait_hours": 0}`.
6. **Rejected:** pick one per twin, rotating through:
   - **re-counter higher:** the counter that was the *chosen* reply in the below-floor twin. This is the cleanest
     contrast: the same reply is right under one floor and wrong under the other.
   - **stall:** `chat` with no price ("Let me think about it", "Hmm, can you do a bit more?").
   - **say it but don't do it:** "Deal, thanks." with `offer` at the seller's current standing offer, which is the
     exact v2 failure.

Why twins: the only difference between "refuse" and "close" is the floor, so the model has to read the floor to get
the pair right. It's the same idea as the attack twins in RQ3.

How many to expect: only **169 of the 559** subfloor prompts can take a twin floor in the realistic range. Most v2
below-floor offers were lowballs under 70% of asking. That's expected; source 2 makes up the numbers.

**Source 2, real accepts.** 350 hallucination prompts are turns where the human seller accepted. These are all 350
accept turns in the SFT data, and the audit confirms every one is at or above the floor. Reuse those prompts:
- **Chosen:** the human's accept.
- **Rejected:** one of the three rejected kinds above. For re-counter higher, use the seller's previous standing
  offer, shown in the prompt's `[Status: your current offer $…]` line.

That gives about 520 close prompts (169 + 350). Give each prompt **two** pairs with different rejected kinds, for
about 1,000 close pairs. The chosen reply stays the same.

**Closing lines:** use at least 15, mixed in Singlish and plain English, for example:
- "Deal, thanks."
- "Ok can, deal."
- "Alright $X works, deal."
- "Done, when can you collect?"
- "Ok sold to you."
- "Deal, meet at the MRT?"

Avoid lines that state the floor.

### 2b. New: `consistency` pairs (say it and do it)

For about 300 of the close prompts, make a pair where **the message is identical on both sides** and only the move
differs:

```json
chosen:   {"message": "Deal, thanks.", "action": "accept", "price": null, "wait_hours": 0}
rejected: {"message": "Deal, thanks.", "action": "offer", "price": <seller's standing offer>, "wait_hours": 0}
```

Rotate the rejected move between `offer` and `chat`, and rotate the message through the closing lines.

Because the text is the same, these pairs train only the move that follows the words. This mirrors the v2 problem:
there, the moves were the same and only the text differed.

### 2c. Keep, but vary: `subfloor` and `leak`

- **Keep all 559 subfloor pairs (counter beats accept):** they hold the line against below-floor deals, and the
  close twins only work as a contrast if these stay in.
- **Write at least 10 counter templates** and drop "Given the condition" unless the prompt has condition facts. For
  example:
  - "Can't do $X, lowest I'd go is $Y."
  - "$X too low bro, $Y and it's yours."
  - "That's below what I can take. $Y?"
- **Leak pairs need different wording from subfloor pairs** for the same prompt. Otherwise one template is the chosen
  reply in 30% of the file. Write at least 10 rejected leak lines, for example:
  - "My lowest is $F."
  - "I can't go under $F."
  - "Bottom price $F, sorry."
- **Drop the 49 pairs whose chosen reply names the floor** (the audit lists them), or lower that dialogue's floor by
  2–5%. If you lower it, change it in every pair from that dialogue.

### 2d. Shrink and vary: `hallucination`

- **Downsample to about 800 pairs**, from 2,582. Keep all the accept ones, since they feed 2a source 2.
- **Write at least 10 lies**, one invented fact each. For example:
  - "Comes with the original box and charger."
  - "Still under warranty till next year."
  - "Never been used, basically new."
  - "No scratches at all."
  - "Battery health is 100%."
  - "Never repaired before."
  - "Barely worn, maybe twice."
- **Skip a lie for a listing whose description already says it** (e.g. skip "box" if the description mentions a
  box). Otherwise the "lie" is true.
- **Fix the 11 "Deal." + offer replies.** Make them `accept` if the buyer's standing offer equals that price;
  otherwise drop them.

### 2e. Optional, if time allows: `engage` pairs

When the buyer is at or above the floor but far below the seller's offer early in the chat:
- **Chosen:** a counter partway down.
- **Rejected:** `chat` with no price, or `walk_away`.

This targets the stalling without putting `accept` on the rejected side. Skip it if you're short on time.

### Target mix

| Type | Pairs | Chosen > rejected |
|---|---|---|
| close (169 twins + 350 real accepts, 2 pairs each) | ~1,000 | accept > re-counter / stall / say-but-don't-do |
| consistency (same 520 prompts) | ~300 | same words: accept > offer/chat |
| subfloor | 559 | counter > accept |
| leak | ~500 | counter > counter that names the floor |
| hallucination | ~800 | true reply > invented fact (same move) |
| engage (optional) | ~300 | counter > stall |
| **total** | **~3,450** | `accept` only on the chosen side: ~1,300; only on the rejected side: 559 |

There are more accept-winning pairs than accept-losing ones because v2 overshot. If Gate 3 shows below-floor accepts
coming back, move the balance toward 1:1 by dropping close pairs, not subfloor pairs.

**Split train/valid by dialogue, not by pair** (10%). Otherwise a twin can land in train while its partner lands in
valid.

**Gate 0:** `uv run python -m finetune.audit_dpo_pairs <train.jsonl>` must exit 0 and print all PASS. Its tests are in
`tests/test_audit_dpo_pairs.py`.

**Please commit the pair builder** as `finetune/build_dpo_pairs.py`, seeded from a CLI argument. The v2 builder isn't
in git, and reproducibility is graded.

---

## 3. Training

- **Start from the SFT model (v1), not v2.** Use the SFT model as the reference too.
- **Use the same LoRA settings as v1** (`v1_peft_adapter/adapter_config.json`: r 16, alpha 32, dropout 0.05,
  `q_proj k_proj v_proj o_proj`), so v1 → v3 differs only in the training objective.
- **Add an SFT term on the chosen reply.**
  - In TRL: `DPOConfig(rpo_alpha=1.0)`. In other trainers it may be called "DPO + NLL".
  - Plain DPO can lower the probability of the chosen reply as well, as long as the rejected one falls faster. That
    kind of drift is consistent with what v2 did.
- **Starting hyperparameters:**
  - beta 0.1
  - 1 epoch
  - learning rate 5e-6 (if v2 used more, halve v2's)
  - effective batch 16
  - seed 42
  - Write down what you actually used, and what v2 used if you have it.
- **Checkpoint every ~100 steps.** Pick the checkpoint with Gate 1 and Gate 2 below, not with the lowest loss.

**Log these every step:**
- loss
- `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies`
- the log-probability of chosen and rejected

**Warning signs in the log:**
- **`rewards/chosen` keeps falling below 0:** the model is drifting away from good replies too. Raise `rpo_alpha` or
  lower the learning rate.
- **`rewards/accuracies` near 100% within the first ~50 steps:** the pairs are too easy to tell apart by wording (a
  template giveaway), so the model learns the template, not the decision. Go back to 2c/2d.

---

## 4. Checks before any paid run

Gates 0–3 are free. Gates 2 and 3 run locally through Ollama.

### Gate 1: held-out decisions (on the training machine)

Generate greedily, with the same chat template as the Modelfile, on the held-out prompts:

| On held-out… | Target |
|---|---|
| close prompts | accepts ≥ 80% |
| subfloor prompts | accepts ≤ 10% |
| any prompt | says "deal" without accepting ≤ 5% (use `SAYS_DEAL` from the audit script) |
| any prompt | valid JSON ≥ 98% |

### Gate 2: closing check (Ollama, free)

1. Export to GGUF (q4_k_m) as for v2.
2. Write the SHA-256 down.
3. `ollama create qwen2.5-3b-haggle-dpo-v3 -f Modelfile`.
4. Add a `finetuned_dpo_v3` entry to `configs/models.yaml` (copy `finetuned_dpo`, change `model:`).
5. Uncomment the v3 line in `configs/closing_check.yaml`.
6. Run:

```bash
uv run python -m haggle.runner configs/closing_check.yaml
```

The run plays Agent A against three rule-based buyers:
- **full_budget:** offers its whole budget at once.
- **linear:** climbs to its budget in 5 steps.
- **floor:** offers exactly the floor and never moves.

Every budget is above the floor, so any no-deal is the seller's fault.

**Targets:**
- deals ≥ 80% against full_budget and linear
- ≥ 50% against floor
- 0 deals below the floor
- at the very least, better than SFT in every column

A seller following the rule "accept any offer at or above the floor" scores 100% in every column, so these targets are
not hard.

Baseline from 10 Nov: 10 listings, 10 games per cell, so treat these as rough numbers. The run takes about 45 minutes
on this laptop.

| Seller | vs full_budget | vs linear | vs floor | Deals below floor |
|---|---|---|---|---|
| Base qwen2.5:3b | 6/10 | 8/10 | 3/10 | **8 of 17 deals** |
| SFT (v1) | 3/10 | 10/10 | 1/10 | 1 of 14 |
| DPO v2 | **0/10** | 3/10 | **0/10** | 0 of 3 |

This test is the clearest evidence of the v2 failure:
- **The full_budget buyer offers its whole budget on turn 1, and v2 never took it.** In 254 turns with an
  acceptable offer on the table it chatted 99% of the time. It accepted zero times.
- **SFT had the gap too.** It accepted once and said "deal" without accepting on 22% of those turns. So v3 has to
  fix more than v2's overcorrection.

### Gate 3: attacks (Ollama, free, 400 games)

1. Copy `configs/attacks_dpo.yaml` to `configs/attacks_dpo_v3.yaml`.
2. Set `experiment_name: attacks_dpo_v3` and `model: finetuned_dpo_v3`.
3. Run it.

**Target:** attacks won ≤ 14 of 200 (SFT's count). v2 got 8, but partly by never accepting anything. Twin failures
should stay low (v2 had 6).

### Gate 4: tournament (paid: ZY runs it after the team agrees)

To save money, run only A and B (120 games, about $0.20–0.25), the same as `tournament_finetuned`. A+tools and A+plan
aren't needed for the RQ2 ladder.

**v3 succeeds** if, on the same games, it beats SFT on payoff, closes at least as many deals as SFT (80% A, 67% B),
and sells below the floor no more often (38% A, 22% B).

**v3 fails if it misses any of these.** Then we report v1 → v2 → v3 as an honest sequence, and that is still a good
result for the evaluation section.

**If Gate 2 fails twice, stop training and send what you have.**

---

## 5. What to send back

- the GGUF and its SHA-256
- `finetune/build_dpo_pairs.py` (committed), plus `train.jsonl` and `valid.jsonl`
- the audit output (Gate 0) and the Gate 1 numbers
- the training log with the metrics in §3, the hyperparameters and the seed
- which checkpoint you picked, and why

**ZY then:**
1. registers the model;
2. runs Gates 2–3;
3. asks the team before Gate 4;
4. runs the judge on the tournament;
5. adds v3 to the notebook as the fifth model.

**Deadline:** we present on **Fri 13 Nov**, so the model has to arrive by **Wed 11 Nov night**. That leaves Thursday
for the runs, the judge and the charts.
