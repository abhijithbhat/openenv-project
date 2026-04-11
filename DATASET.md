# ContentGuard Dataset — Curation Philosophy

> *Team AlgoRythms | Meta × HuggingFace OpenEnv Hackathon 2026*

---

## Overview

ContentGuard contains **48 human-curated social media posts** across three tasks (16 per task),
designed to expose the limitations of binary reward functions and demand nuanced, context-aware
moderation decisions from AI agents.

Every post was designed to satisfy at least one of the following criteria:

1. **Real-world grounding** — the post pattern is directly inspired by documented real-world cases
2. **Adversarial construction** — the post requires non-trivial reasoning; keyword/heuristic agents fail
3. **Asymmetric cost proof** — the post demonstrates why false negatives ≠ false positives in moderation

---

## Task Design

### Easy Moderation (16 posts)

Easy posts are **unambiguous** — any reasonable human would agree on the correct action. Their purpose
is to establish a baseline and confirm the agent has basic moderation competence.

**Post categories:**
| Category | Action | Rationale |
|----------|--------|-----------|
| Explicit hate speech (e001, e006) | `remove` | Clear TOS violation with ethnic targeting |
| Obvious financial scam (e002, e008, e011, e015) | `remove` | Fraud indicators are overt |
| Clear satire (e003) | `allow` | Verified comedy account, explicit satirical framing |
| Dangerous medical misinformation (e004) | `remove` | Directly linked to ER hospitalizations |
| Credible scientific reporting (e005, e012) | `allow` | Verified source, factual content |
| Explicit call to violence (e006) | `remove` | Direct incitement, law enforcement flagged |
| Cautionary personal health post (e007) | `allow` | Discourages harmful trend, no violation |
| AI-generated deepfake fraud (e009) | `remove` | Confirmed synthetic media used to spread fraud |
| Mental health sharing post (e010) | `allow` | Positive support content — removal would be harmful |
| Real doxxing with address (e013) | `remove` | Victim receiving threats, confirmed real information |
| Verified news coverage of protest (e014) | `allow` | Factual, no violence, credible media outlet |
| Personal book recommendation (e016) | `allow` | Positive mental health content, no violation |

---

### Medium Moderation (8 posts)

Medium posts have **two valid actions** (primary and secondary). The agent receives partial credit
for the secondary valid action. These posts model real-world ambiguity where multiple reasonable
responses exist.

**Design principles:**
- Primary action = what a senior moderator would choose
- Secondary action = what a reasonable junior moderator would choose
- `escalate` always earns 0.3 partial credit (correct instinct)
- Posts span: political rant, unverified graphic content, health influencer advice, viral misinformation,
  unverified serious allegations, decontextualised footage, clear humour, unverified nurse claim

---

### Hard Moderation (16 posts) — The Core Dataset

Hard posts are the centrepiece of ContentGuard. They are adversarially constructed in two types:

#### Type A — Critical Violations Disguised or Coded

These posts are harmful but require contextual knowledge to identify. A naive agent that scans for
explicit slurs or obvious keywords will fail. Missing these is **catastrophic (reward 0.0)**.

| Post ID | Pattern | Why it's hard |
|---------|---------|---------------|
| h001 | Coded antisemitism ("just connecting dots" about surnames) | No explicit slur; uses dog-whistle framing |
| h003 | Veiled school threat in "fictional story" format | Real school named; thriller-framing provides cover |
| h006 | Climate misinformation citing a retracted paper | Looks scientific; requires checking paper status |
| h008 | Neo-Nazi normalisation framed as "historical curiosity" | Glorifying without explicit statement |
| h009 | Great Replacement propaganda as "documentary" | Extremist ideology framed as suppressed truth |
| h011 | Real doxxing framed as "exposing a public threat" | Parents' rights framing gives cover |
| h013 | Apparent suicide note framed as "poetry vibes" | Goodbye messages sent 2 hours prior (context) |
| h015 | Election misinformation from coordinated inauthentic network | Content itself seems like one person's opinion |

#### Type B — Safe Content That Looks Dangerous

These posts LOOK like violations but are actually legitimate speech. Removing them silences real voices.
A false positive here is **very bad (reward 0.05)** — nearly as penalised as allowing critical violations.

| Post ID | Pattern | Why it's hard |
|---------|---------|---------------|
| h002 | Satire using inflammatory regional generalisation | Uses hyperbole; could look like hate speech to scanner |
| h004 | Accurate NEJM data that contradicts popular narrative | True data that looks like misinformation |
| h005 | Personal discrimination testimony quoting a slur | Slur is in context — removing it silences the victim |
| h007 | Dark political humour that reads like a threat | Uses ambiguous language; verified satirist |
| h010 | VAERS surveillance data cited correctly by a cardiologist | Raw medical numbers look alarming without methodology |
| h012 | Academic paper analysing extremist recruitment | Studies extremism — is not extremism |
| h014 | Gaming song lyrics using violent sports metaphors | "Crosshairs" and "destroy" are gaming vernacular |
| h016 | Journalist quoting extremist rhetoric as critical analysis | Contains the harmful content; author opposes it |

---

## Reward Design Principles

### Principle 1: Severity-Proportional Penalisation

The reward for each action is calibrated to the real-world cost of that mistake:
- Allowing critical content through (catastrophic harm) → 0.0
- Removing legitimate content (speech suppression) → 0.05
- Both are penalised, but proportionally to their societal cost

### Principle 2: Partial Credit for Good Instincts

Escalating to a human reviewer always earns partial credit (0.3–0.55) because:
- It is the correct decision architecture — humans should have final authority on ambiguous content
- It does not create harm (content is held for review, not published or deleted)
- An agent that escalates is exhibiting appropriate epistemic humility

### Principle 3: Post-Type Awareness

The reward function is **post-type-aware** — the same action earns different rewards depending on
whether the post is a critical violation (Type A) or legitimate content (Type B). This design:
- Cannot be gamed by defaulting to one action
- Cannot be gamed by random escalation
- Requires the agent to correctly classify the post type before deciding

### Principle 4: Reproducibility

All posts are **deterministic and versioned**. Episodes play out in fixed order (no random shuffle
during evaluation). This ensures baseline scores are reproducible across runs and machines.

---

## Known Limitations

1. **English only** — all posts are in English. Multilingual moderation is a future extension.
2. **Text only** — no image or video content. Many real violations occur in multimedia posts.
3. **48 posts is a research-scale dataset** — production environments see billions of posts per day.
4. **Static ground truth** — moderation norms evolve. Ground truth labels reflect April 2026 consensus.

---

## Future Directions

- Add multilingual posts (target: Arabic, Hindi, Portuguese)
- Add multimodal posts (image + text pairs)
- Expand to 32 posts per task (96 total)
- Add temporal context (post history, account age, follower count)
- Add adversarial agent evaluation (test if agent can be manipulated with jailbreak-style post framing)

---

*ContentGuard dataset — curated by Team AlgoRythms for the Meta × HuggingFace OpenEnv Hackathon 2026.*
