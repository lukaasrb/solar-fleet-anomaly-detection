# Solar Fleet Anomaly Detection

Unsupervised battery-fault detection for a fleet of solar-powered IoT devices — no labels
required to start, no black box once it's running. Every alert is explainable in volts and
days: *which* rule fired, *which other* rules fired alongside it, and *why* that rule's
threshold is what it is. On top of that sits an LLM layer that turns an alert into a field
briefing for a technician — and every number it writes is audited back against the
engine's own measurements.

This is a clean-room reimplementation of production fleet-monitoring techniques, rebuilt
from scratch against a synthetic dataset so the methodology can be shared publicly. The
synthetic generator (`src/generate_synthetic_fleet.py`) is the only source of data in this
repository — no real device, company, or telemetry data anywhere.

## Results

Measured against synthetic ground truth (200 devices, 240 days, 37 injected fault episodes
across 6 distinct fault types, plus 16 devices deliberately built to fool a naive detector):

| Metric | Value |
|---|---|
| Precision | **94.7%** |
| Recall | **86.5%** |
| False-positive traps correctly avoided | **16 / 16** |
| Daily vs. batch engine agreement | **0 mismatches** over 40 days |

Reproduce these numbers yourself:

```
pip install -r requirements.txt
python src/generate_synthetic_fleet.py
python src/validate.py
```

`validate.py` runs the full pipeline twice — once cold, once after simulating a human
reviewing exactly the episodes the model already flagged — and reports precision/recall
before and after, plus a per-channel and per-fault-type breakdown. Nothing here is hand-picked
to look good: this is the actual output of the actual code, on data anyone can regenerate.

## The problem this is solving

A fleet of battery-backed devices reports daily telemetry. Most devices are healthy. A few
are failing in different ways — a catastrophic short, a panel that's stopped recharging, a
sensor that's frozen, a young device dying before it ever established a baseline. There are
no fault labels to start from: nobody has manually reviewed years of history to say "this day
was a real fault." The detector has to work from telemetry alone, and it has to explain
itself well enough that a technician trusts the alert enough to act on it.

## Six channels, not one score

| Channel | Catches | Key idea |
|---|---|---|
| `SHOCK` | Catastrophic voltage collapse | Absolute threshold, immediate — a real crash doesn't wait for statistical persistence |
| `UNDERCHARGE` | Panel stopped recharging | **Self-referential**: compares the device to its *own* history, not its neighbors' |
| `CHRONIC_LEVEL` | Persistent below-normal level | Cross-sectional Z-score vs. cluster, **gated** by the device's own baseline too |
| `FLATLINE` | Frozen sensor | Near-zero daily oscillation — excluding a battery that's simply full |
| `MONOTONIC_ONSET` | A young device dying from day one | Decline from a *local* peak (resets on any recovery), not "ever below its best day" |
| `RECURRING_SHUTDOWN` | Repeated die-and-revive cycles | The revival is always *higher* than before it died — invisible to any rule that expects a degraded return |

A day can trigger more than one channel — in this dataset, roughly a quarter of anomalous
days do. Each detected day gets a **priority label** (the single most actionable channel, for
filtering/routing) and a **full list of every channel that fired** (for understanding the
complete picture) — collapsing that to one label per day would silently discard real signal.

## Three engineering decisions worth reading the code for

**1. Self-referential beats cross-sectional, and here's the proof.** `CHRONIC_LEVEL`
compares a device to its cluster peers *on that day* — the obvious first approach, and the
one most naive detectors stop at. On its own it permanently false-alarms on any device that's
structurally different from its neighbors but stable over time (different panel orientation,
local shading — healthy, just not average). This repo ships 16 such devices specifically to
prove the point: `CHRONIC_LEVEL` is gated on the device's *own* historical floor as well as
the cross-sectional Z-score, and the trap devices are correctly never flagged —
`0 / 16` in the validation report, reproducible from a clean checkout.

**2. A missing day is not a bad day.** Consecutive-day persistence logic (`UNDERCHARGE`
needs 5 straight bad days) breaks around network gaps if you compute it naively over a
calendar grid — depending on which direction the condition runs, a gap either resets a real
streak (hiding an active fault) or inflates one (raising a false alarm). See
`consecutive_run_ignoring_gaps()`: gap days are invisible to the streak, not zero and not one.

**3. Baseline decontamination.** A device's "what's normal for me" statistic — the number
every self-referential channel is compared against — was, in an earlier version of this
logic, computed over its *entire* history, fault days included. A confirmed fault then
dilutes its own reference point, becoming harder to detect (or making the next one harder).
Once a human confirms an episode as real, that window is excluded from every future baseline
calculation — see `decontaminated_median()` and the `confirmed_fault_episodes` parameter
threaded through `train_pipeline.run_pipeline()`. The effect scales with how much of a
device's *observed history* a fault represents: negligible for a device with years of mostly
healthy data, large for one whose short life is mostly the fault itself (`validate.py`
includes a deliberately constructed device of the second kind — see `DEV00001` in its output).
Note what this **isn't**: at no point does the pipeline see the synthetic ground truth
directly. `validate.py` simulates the real workflow — a human reviewing only what the model
already surfaced — never an oracle peek at the full fault list.

## Two-speed architecture

| Module | Cadence | Cost |
|---|---|---|
| `train_pipeline.py` | Weekly / offline | Refits clustering, re-freezes every baseline, rescans full history |
| `daily_inference.py` | Daily / online | Scores a 45-day rolling window using the frozen artifacts — cost stays flat as history grows |

The daily module never refits anything; it only *applies* what the weekly run froze. In this
demo it reuses the batch engine's channel logic directly (see the module docstring in
`daily_inference.py` for why a real production deployment might instead deliberately
duplicate that logic into a fully separate module — a genuine trade-off between DRY-ness and
isolating an in-flight weekly refactor from an already-running daily job).

## The technician assistant: an LLM that explains, never detects

`technician_assistant.py` turns a detected episode into a field briefing — what is physically
wrong, what to load into the van, whether to drive out today. It is the only place in the
repository where a language model is involved, and the boundary is drawn deliberately:

| Layer | Who produces it | What it produces |
|---|---|---|
| **Evidence** (`build_evidence`) | Deterministic code | Every number: the voltage that crossed a threshold, the margin, the device's own frozen baseline, the streak length, the silent days |
| **Briefing** (`brief_episode`) | Claude, structured output | Only *language*: explanation, likely physical cause, recommended action, urgency, parts list |
| **Audit** (`audit_numbers`) | Deterministic code | Every numeric token in the model's prose, checked back against the evidence packet |

The model is never asked *whether* a device is faulty or *why a channel fired* — the engine
already computed both, and the reasoning is handed over as finished sentences. Its job is
translation and field judgement.

**The third row is the point.** A prompt instructing a model not to invent figures is a hope;
a technician acting on a hallucinated voltage is a wasted truck roll. So every number in the
output is checked against the packet and unsupported ones are reported as findings:

```
grounded briefing    -> clean
fabricated briefing  -> ['13.9', '4.7', '72']
```

**It runs without an API key.** Missing SDK, missing credentials, API error, or a refusal all
fall back to `_deterministic_briefing()`, which renders the same four fields from the same
evidence packet using the channel playbook. A clean checkout produces useful output — clearly
labelled as the template path — and the daily job never has a hard dependency on an external
service being reachable.

```
python src/technician_assistant.py --limit 5     # uses Claude if credentials are available
python src/technician_assistant.py --offline     # never contacts the API
```

Two further decisions worth noting: the static half of the prompt (rules plus the channel
playbook) sits behind a cache breakpoint, so a run over many episodes pays for it once; and
briefings are issued one request at a time, which suits a daily run over a handful of new
episodes. A fleet-wide backfill over months of history should use the Message Batches API
instead — same prompts, half the price — documented rather than implemented, because the
batch path only pays off at a volume this demo fleet never reaches.

## Architecture notes

- **Train/Test split for clustering** (`TRAIN_CUTOFF_DAY`), never for the rule engine itself
  — clustering and baselines are fit on the Train period only, so precision/recall on the
  Test period are an honest, out-of-time estimate.
- **Cluster safety net** (`assign_fallback_clusters`): a device with no telemetry before the
  Train cutoff — onboarded late, or barely producing clean readings because it's already
  failing — never enters the clustering fit. Without a fallback it silently disappears from
  every downstream table, fault included. Falls back to the fleet's most common cluster
  rather than dropping the device.
- **Episode construction**: consecutive anomalous days collapse into one episode (small gaps
  don't split it) — this is how the output should actually be read, one incident per row,
  not one row per day.

## Project layout

```
src/
  generate_synthetic_fleet.py   only source of data — 200 devices, 6 fault types, 16 traps
  train_pipeline.py             weekly: clustering, decontaminated baselines, 6-channel engine
  daily_inference.py            daily: rolling-window scoring from frozen artifacts
  validate.py                   precision/recall vs. ground truth, before/after decontamination
  technician_assistant.py       episode -> field briefing (Claude), with a numeric audit
                                and a deterministic fallback that needs no API key
data/        generated locally, gitignored
models/      frozen artifacts (scaler, kmeans, baselines), gitignored
```

## Where this stops, and why

Every number above is reproducible from a clean clone, which means every weakness is too.
Four are worth stating outright, with what each would actually take to close.

**`RECURRING_SHUTDOWN` catches 3 of its 6 injected episodes** — the weakest channel by a
clear margin; the next weakest, `UNDERCHARGE`, catches 5 of 7. Its detection window keys on
the spacing of die/revive cycles, so a device cycling faster or slower than that window slips
past it. The fix is not a threshold tweak. It needs a wider sample of real cycle geometries
before the window is worth tuning at all — and manufacturing that sample synthetically would
only prove the channel can find what it was shaped to find.

**Cluster count is fixed at `N_CLUSTERS = 4`, not chosen from the data.** It matches the four
climate regions the generator builds in, which makes it correct for this fleet and useless as
a general answer. A real deployment wants silhouette or gap-statistic selection. It is absent
here for a specific reason: a demo that ships its own ground truth would be picking *k*
against data whose structure it already knows, which proves nothing.

**No thermal/humidity safety channel, no dashboard.** This is the detection engine plus its
explanation layer, scoped to stay readable in one sitting rather than reproducing an
operations platform. Both fields are already in the telemetry schema — what is missing is
scope, not difficulty.

**The briefings are not graded against field outcomes.** The numeric audit establishes that
they are *grounded*: no figure reaches a technician that the engine did not measure. It
establishes nothing about whether "bring a charge controller" was the right call, and no
synthetic fleet can — there is no ground truth for what a technician found on site. Before
the urgency labels are trusted operationally, a deployment needs technicians scoring a sample
of real briefings against what the repair actually turned out to be.

## License

MIT — see `LICENSE`.
