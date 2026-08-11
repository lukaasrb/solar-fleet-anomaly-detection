"""
LLM-backed technician assistant: turns a detected episode into a field briefing.

The detection engine (train_pipeline.py) answers "which device, which day, which channel".
A technician needs something else: what is physically wrong, what to bring, and whether to
drive out today. This module closes that gap - and it is deliberately the *only* place in
the repository where a language model is involved.

The division of labour is the whole design:

  DETERMINISTIC (build_evidence)   Every number a technician might act on - the voltage that
                                   crossed a threshold, the margin by which it crossed, the
                                   device's own frozen baseline, the streak length, the gap
                                   count - is computed from the telemetry and the frozen
                                   artifacts. Nothing here is inferred.

  GENERATIVE (brief_episode)       The model receives that evidence packet and produces only
                                   *language*: an explanation, a likely physical cause, a
                                   recommended action, an urgency label. It is instructed to
                                   quote the supplied numbers and invent none.

  AUDITED (audit_numbers)          Every numeric token in the model's prose is checked back
                                   against the evidence packet. A number that appears in the
                                   briefing but not in the evidence is reported as a finding,
                                   not silently trusted. This is what makes the previous
                                   sentence a property of the system rather than a promise in
                                   a prompt.

Runs without an API key. `_deterministic_briefing()` renders the same four fields from the
same evidence packet using the channel playbook below, so a clean checkout produces useful
output - degraded, clearly labelled, but never a crash or an empty page. The LLM path is an
upgrade to the wording, not a dependency of the pipeline.

Cost note: this briefs episodes one request at a time, which is the right shape for a daily
run (a handful of new episodes, results wanted immediately). A fleet-wide backfill over
months of history should use the Message Batches API instead - same prompts, 50% of the
price, results within the hour. Documented rather than implemented: the batch path only pays
off at a volume this demo fleet never reaches.
"""
import argparse
import os
import re
import textwrap
from dataclasses import dataclass, field

import pandas as pd

import train_pipeline as tp

MODEL = "claude-opus-5"
MAX_TOKENS = 16000


# ==========================================================================================
# Channel playbook - deterministic domain reference
# ==========================================================================================
# Physical meaning and default field response per channel. This is engineering knowledge
# about the fleet, not model output: it is fed to the model as context, AND used directly by
# the no-API-key fallback. One source of truth for both paths.
CHANNEL_PLAYBOOK = {
    "SHOCK": {
        "physical": "Catastrophic loss of cell voltage - an internal short, a disconnected "
                    "cell string, or a battery that has failed outright.",
        "action": "Dispatch. The battery is not recoverable in place; bring a replacement pack.",
        "urgency": "dispatch_now",
        "parts": ["replacement battery pack", "terminal hardware"],
    },
    "UNDERCHARGE": {
        "physical": "The panel is no longer replenishing the battery: shading, soiling, a "
                    "failed charge controller, or a broken panel-to-controller connection. "
                    "The battery itself is usually still healthy at this stage.",
        "action": "Schedule a visit. Inspect panel surface, tilt and wiring before condemning "
                  "the battery - most of these are a controller or a connector, not a cell.",
        "urgency": "schedule_visit",
        "parts": ["charge controller", "panel cleaning kit", "multimeter"],
    },
    "CHRONIC_LEVEL": {
        "physical": "The device sits persistently below both its own historical floor and its "
                    "cluster peers - consistent with capacity fade from ageing cells or a "
                    "parasitic load that never turns off.",
        "action": "Schedule a visit. Measure standby current draw first; if it is nominal, the "
                  "cells have aged and the pack is due for replacement.",
        "urgency": "schedule_visit",
        "parts": ["clamp meter", "replacement battery pack"],
    },
    "FLATLINE": {
        "physical": "The reported voltage has stopped moving. A real battery always oscillates "
                    "between charge and discharge - a frozen reading is a sensor or telemetry "
                    "fault, not a battery fault.",
        "action": "Remote first: power-cycle the telemetry unit before dispatching anyone. "
                  "The battery is probably fine and the reading is not to be trusted.",
        "urgency": "monitor",
        "parts": ["spare telemetry module"],
    },
    "MONOTONIC_ONSET": {
        "physical": "A device declining from the day it was commissioned and never recovering "
                    "- an install fault or a pack that shipped defective, not degradation.",
        "action": "Dispatch and treat as a warranty case. Check install polarity and panel "
                  "orientation before replacing the pack; capture the readings for the claim.",
        "urgency": "dispatch_now",
        "parts": ["replacement battery pack", "install checklist", "camera"],
    },
    "RECURRING_SHUTDOWN": {
        "physical": "Repeated die-and-revive cycles: the device drains, shuts down, recharges "
                    "at zero load, and restarts. Undersized generation for the actual duty "
                    "cycle, or a load that exceeds what the panel was specified for.",
        "action": "Schedule a visit and review the sizing, not just the hardware. Replacing "
                  "the pack without changing the panel or the duty cycle repeats the fault.",
        "urgency": "schedule_visit",
        "parts": ["larger panel", "load profile logger"],
    },
}


# ==========================================================================================
# Evidence packet - 100% deterministic
# ==========================================================================================
@dataclass
class EpisodeEvidence:
    """Everything a briefing is allowed to be based on. Every field is measured or read from
    a frozen artifact; none is inferred."""
    sn: str
    cluster: int
    start_day: int
    end_day: int
    span_days: int
    real_days: int
    gap_days: int
    device_age_at_onset: int
    priority_channel: str
    channels_fired: list
    battery_min_low: float
    battery_min_median: float
    battery_max_median: float
    charge_delta_median: float
    baseline_battery_min: float
    baseline_battery_max: float
    baseline_charge_delta: float
    cluster_median_battery_min: float
    findings: list = field(default_factory=list)

    def as_prompt_block(self):
        """Render the packet for the model. This exact text is also what audit_numbers()
        checks the model's output against - one representation, no drift."""
        lines = [
            f"DEVICE: {self.sn}   CLUSTER: {self.cluster}",
            f"EPISODE: day {self.start_day} to {self.end_day} "
            f"({self.span_days} calendar days, {self.real_days} with telemetry, "
            f"{self.gap_days} silent)",
            f"DEVICE AGE AT ONSET: {self.device_age_at_onset} days",
            f"PRIORITY CHANNEL: {self.priority_channel}",
            f"ALL CHANNELS FIRED: {', '.join(self.channels_fired)}",
            "",
            "MEASUREMENTS DURING THE EPISODE",
            f"  lowest battery_min      {self.battery_min_low} V",
            f"  median battery_min      {self.battery_min_median} V",
            f"  median battery_max      {self.battery_max_median} V",
            f"  median daily swing      {self.charge_delta_median} V",
            "",
            "THIS DEVICE'S OWN FROZEN BASELINE (healthy history, fault days excluded)",
            f"  battery_min baseline    {self.baseline_battery_min} V",
            f"  battery_max baseline    {self.baseline_battery_max} V",
            f"  daily swing baseline    {self.baseline_charge_delta} V",
            "",
            f"CLUSTER PEERS, SAME DAYS: median battery_min {self.cluster_median_battery_min} V",
            "",
            "WHY EACH CHANNEL FIRED",
        ]
        lines += [f"  - {f}" for f in self.findings]
        return "\n".join(lines)


def _fmt(x, nd=2):
    return round(float(x), nd)


def _channel_findings(channel, ev):
    """One deterministic sentence per fired channel, stating the measured value, the threshold
    it crossed, and the margin. These are the sentences the model is asked to translate - it
    is never asked to work out *why* a channel fired."""
    if channel == "SHOCK":
        return (f"SHOCK: battery_min reached {ev.battery_min_low} V, below the absolute "
                f"shock floor of {tp.SHOCK_ABSOLUTE_FLOOR} V "
                f"(margin {_fmt(tp.SHOCK_ABSOLUTE_FLOOR - ev.battery_min_low)} V), sustained "
                f"at least {tp.SHOCK_PERSISTENCE_DAYS} days. This threshold is absolute and "
                f"cluster-independent - a real collapse is not made more certain by waiting.")
    if channel == "UNDERCHARGE":
        floor = _fmt(ev.baseline_battery_max - tp.UNDERCHARGE_MARGIN)
        return (f"UNDERCHARGE: median battery_max {ev.battery_max_median} V against this "
                f"device's own recharge baseline of {ev.baseline_battery_max} V - below its "
                f"personal floor of {floor} V (baseline minus the "
                f"{tp.UNDERCHARGE_MARGIN} V margin) for at least "
                f"{tp.UNDERCHARGE_MIN_STREAK} consecutive days that actually reported. "
                f"Compared to itself, not to the fleet.")
    if channel == "CHRONIC_LEVEL":
        return (f"CHRONIC_LEVEL: median battery_min {ev.battery_min_median} V sits below both "
                f"this device's own floor of {ev.baseline_battery_min} V (by more than the "
                f"{tp.CHRONIC_SELF_MARGIN} V gate) and its cluster's median of "
                f"{ev.cluster_median_battery_min} V (Z below {tp.CHRONIC_Z_THRESHOLD}), for "
                f"at least {tp.CHRONIC_PERSISTENCE_DAYS} days. Both gates are required - the "
                f"peer comparison alone would flag healthy devices that are simply different.")
    if channel == "FLATLINE":
        return (f"FLATLINE: median daily swing {ev.charge_delta_median} V, under the "
                f"{tp.FLATLINE_NOISE_FLOOR} V noise floor, for at least "
                f"{tp.FLATLINE_MIN_STREAK} days, while NOT sitting near this device's "
                f"full-charge ceiling - a saturated battery is excluded, so this is a stuck "
                f"reading rather than a full one.")
    if channel == "MONOTONIC_ONSET":
        return (f"MONOTONIC_ONSET: battery_max fell at least "
                f"{tp.MONOTONIC_CUMULATIVE_DROP} V from its local peak across "
                f"{tp.MONOTONIC_MIN_STREAK}+ days without a single recovery, on a device only "
                f"{ev.device_age_at_onset} days old (young-device window: "
                f"{tp.YOUNG_DEVICE_MAX_AGE_DAYS} days). The peak resets on any recovery, so "
                f"this is a genuine one-way decline, not a noisy series below its best day.")
    if channel == "RECURRING_SHUTDOWN":
        return (f"RECURRING_SHUTDOWN: at least {tp.SHUTDOWN_CYCLES_THRESHOLD} silence gaps of "
                f"{tp.SHUTDOWN_GAP_MIN}-{tp.SHUTDOWN_GAP_MAX} days within a "
                f"{tp.SHUTDOWN_WINDOW_DAYS}-day window. Each return is HIGHER than before the "
                f"gap, because the panel recharges at zero load while the device is off - "
                f"which is why a rule expecting a degraded return never sees this.")
    return f"{channel}: fired."


def build_evidence(result, episodes, baselines):
    """Assemble one evidence packet per detected episode. Pure measurement - no model."""
    baseline_by_sn = baselines.set_index("sn")
    result = result.copy()
    result["charge_delta"] = result["battery_max"] - result["battery_min"]

    # Recomputed exactly as run_engine does it (same rows, same grouping), so the peer figure
    # quoted to the technician is the one the engine actually gated on.
    cluster_median = (result.groupby(["day_index", "cluster"])["battery_min"]
                      .median().rename("cluster_median").reset_index())
    result = result.merge(cluster_median, on=["day_index", "cluster"], how="left")

    packets = []
    for row in episodes.itertuples(index=False):
        window = result[(result["sn"] == row.sn) &
                        (result["day_index"] >= row.start_day) &
                        (result["day_index"] <= row.end_day)]
        if window.empty:
            continue
        flagged = window[window["is_anomalous"] == 1]
        if flagged.empty:
            continue

        channels = sorted({c for joined in flagged["active_channels"]
                           for c in joined.split("+") if c != "NONE"})
        priority = flagged["primary_channel"].value_counts().idxmax()
        span = int(row.end_day - row.start_day + 1)
        first_seen = int(baseline_by_sn.loc[row.sn, "first_seen_day"])

        ev = EpisodeEvidence(
            sn=row.sn,
            cluster=int(window["cluster"].iloc[0]),
            start_day=int(row.start_day),
            end_day=int(row.end_day),
            span_days=span,
            real_days=int(len(window)),
            gap_days=int(span - len(window)),
            device_age_at_onset=int(row.start_day - first_seen),
            priority_channel=priority,
            channels_fired=channels,
            battery_min_low=_fmt(window["battery_min"].min()),
            battery_min_median=_fmt(window["battery_min"].median()),
            battery_max_median=_fmt(window["battery_max"].median()),
            charge_delta_median=_fmt(window["charge_delta"].median()),
            baseline_battery_min=_fmt(baseline_by_sn.loc[row.sn, "battery_min_baseline"]),
            baseline_battery_max=_fmt(baseline_by_sn.loc[row.sn, "battery_max_baseline"]),
            baseline_charge_delta=_fmt(baseline_by_sn.loc[row.sn, "charge_delta_baseline"]),
            cluster_median_battery_min=_fmt(window["cluster_median"].median()),
        )
        ev.findings = [_channel_findings(c, ev) for c in channels]
        packets.append(ev)
    return packets


# ==========================================================================================
# Generative layer
# ==========================================================================================
SYSTEM_PROMPT = """\
You write field briefings for technicians maintaining a fleet of solar-powered, \
battery-backed IoT devices. A statistical detection engine has already decided that a device \
is faulty and has computed every relevant measurement. Your job is to turn that evidence into \
something a technician can act on before they get in the van.

THE ONE RULE: every number you write must appear verbatim in the evidence packet you are \
given. Do not calculate new figures, do not estimate, do not round differently, do not \
introduce a threshold, voltage, duration or count that is not in the packet. If you want to \
express a magnitude the packet does not contain, describe it in words instead. Your output is \
checked against the packet automatically and unsupported numbers are reported as defects.

You are not the detector. Do not re-litigate whether the device is faulty, do not suggest the \
alert may be spurious, and do not propose further statistical analysis. The engine gates every \
channel on the device's own history precisely so that structurally unusual but healthy devices \
are never flagged - treat the detection as sound and move to what happens next.

Write for someone competent and busy. The summary is one sentence a dispatcher can read at a \
glance. The likely cause names the physical failure, and says plainly when the evidence \
distinguishes a battery fault from a charging fault from a sensor fault - that distinction \
determines what gets loaded into the van. The recommended action is what to do first on site, \
including any check that could avoid replacing a part unnecessarily.

Choose urgency honestly. dispatch_now means the device is dead or dying today. \
schedule_visit means it is degrading on a timescale of weeks. monitor means the evidence \
points at telemetry rather than hardware, or the device is stable enough to watch.

Set confidence to low when channels disagree, when the episode is short, or when a large \
share of the window is silent; high only when the fired channels tell one coherent story."""


def _build_client():
    """Returns an Anthropic client, or None if the SDK or a credential is unavailable.

    An unset ANTHROPIC_API_KEY does not by itself mean there are no credentials - the SDK also
    resolves an `ant auth login` profile - so construction is attempted regardless and only a
    genuine failure disables the LLM path.
    """
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic()
    except Exception:
        return None


def _briefing_model():
    """The structured-output schema. Defined lazily so the module imports without pydantic."""
    from typing import Literal

    from pydantic import BaseModel, Field

    class TechnicianBriefing(BaseModel):
        summary: str = Field(description="One sentence a dispatcher can read at a glance.")
        likely_cause: str = Field(description="The physical failure, and what it is not.")
        recommended_action: str = Field(description="What to do first on site.")
        urgency: Literal["dispatch_now", "schedule_visit", "monitor"]
        parts_to_bring: list[str]
        confidence: Literal["high", "medium", "low"]

    return TechnicianBriefing


def brief_episode(client, evidence):
    """One episode -> one structured briefing. Returns (briefing_dict, source_label).

    Falls back to the deterministic briefing on any API failure or refusal: a technician
    getting the template wording is a far better outcome than a daily job that dies because
    an upstream service was unavailable.
    """
    import anthropic

    TechnicianBriefing = _briefing_model()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # The playbook and the rules are identical for every episode in a run, so they sit
            # in a cached prefix; only the evidence packet below varies per request.
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\nCHANNEL REFERENCE\n" + _playbook_text(),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": evidence.as_prompt_block()}],
            output_format=TechnicianBriefing,
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        return _deterministic_briefing(evidence), f"fallback (API error: {type(exc).__name__})"

    # stop_reason must be checked before the payload is read - a refused or truncated turn
    # leaves parsed_output unpopulated.
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return _deterministic_briefing(evidence), "fallback (no usable model output)"
    return response.parsed_output.model_dump(), "claude"


def _playbook_text():
    out = []
    for channel, entry in CHANNEL_PLAYBOOK.items():
        out.append(f"{channel}: {entry['physical']} Typical response: {entry['action']}")
    return "\n".join(out)


def _deterministic_briefing(evidence):
    """The no-API-key path. Same evidence, same playbook, template wording.

    This is what makes the module runnable from a clean checkout - and it is also the failure
    mode for every LLM error path, so the daily job never has a hard dependency on an external
    service being up.
    """
    entry = CHANNEL_PLAYBOOK[evidence.priority_channel]
    others = [c for c in evidence.channels_fired if c != evidence.priority_channel]
    also = f" {len(others)} further channel(s) fired: {', '.join(others)}." if others else ""
    return {
        "summary": f"{evidence.sn}: {evidence.priority_channel} over "
                   f"{evidence.span_days} days from day {evidence.start_day}.{also}",
        "likely_cause": entry["physical"],
        "recommended_action": entry["action"],
        "urgency": entry["urgency"],
        "parts_to_bring": list(entry["parts"]),
        "confidence": "high" if len(evidence.channels_fired) > 1 else "medium",
    }


# ==========================================================================================
# Numeric audit - the claim, enforced
# ==========================================================================================
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def audit_numbers(briefing, evidence):
    """Every number in the briefing prose must appear in the evidence packet.

    Returns the list of unsupported numbers - empty means the briefing is fully grounded. This
    is deliberately checked rather than merely instructed: a prompt asking a model not to
    invent figures is a hope, and a technician acting on a hallucinated voltage is a truck roll.

    Both sides are compared as normalised numeric strings, so 8.0 and 8 match. Reported, not
    raised: an unsupported number is a finding for the operator to see, and suppressing the
    whole briefing over one stray digit would help nobody.
    """
    def norm(tokens):
        out = set()
        for t in tokens:
            try:
                v = float(t)
            except ValueError:
                continue
            out.add(f"{v:g}")
        return out

    allowed = norm(_NUMBER.findall(evidence.as_prompt_block()))
    prose = " ".join([briefing["summary"], briefing["likely_cause"],
                      briefing["recommended_action"], " ".join(briefing["parts_to_bring"])])
    return sorted(norm(_NUMBER.findall(prose)) - allowed)


# ==========================================================================================
# CLI
# ==========================================================================================
def _row(label, text):
    """Label in the left gutter, body wrapped and hanging-indented under itself."""
    body = textwrap.fill(str(text), width=96, subsequent_indent=" " * 12)
    print(f"  {label:<9} {body}")


def render(briefing, evidence, source, unsupported):
    print("=" * 108)
    print(f" {evidence.sn}  |  days {evidence.start_day}-{evidence.end_day}  |  "
          f"{evidence.priority_channel}  |  {'+'.join(evidence.channels_fired)}")
    print("=" * 108)
    _row("URGENCY", f"{briefing['urgency'].upper()}   "
                    f"(confidence: {briefing['confidence']}, written by: {source})")
    _row("SUMMARY", briefing["summary"])
    _row("CAUSE", briefing["likely_cause"])
    _row("ACTION", briefing["recommended_action"])
    _row("BRING", ", ".join(briefing["parts_to_bring"]))
    if unsupported:
        _row("! AUDIT", "numbers not present in the evidence packet: "
                        + ", ".join(unsupported))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--limit", type=int, default=5,
                        help="how many episodes to brief (default 5)")
    parser.add_argument("--offline", action="store_true",
                        help="skip the model entirely and use the deterministic path")
    args = parser.parse_args()

    result, episodes = tp.run_pipeline()
    baselines = pd.read_pickle(tp.MODELS_DIR / "device_baselines.pkl")
    packets = build_evidence(result, episodes, baselines)
    print(f"\n{len(packets)} episodes with evidence; briefing {min(args.limit, len(packets))}.\n")

    client = None if args.offline else _build_client()
    if client is None and not args.offline:
        print("No Anthropic client available (SDK missing or no credentials) - "
              "using the deterministic path.\n")

    audited = flagged = 0
    for evidence in packets[:args.limit]:
        if client is None:
            briefing, source = _deterministic_briefing(evidence), "deterministic template"
        else:
            briefing, source = brief_episode(client, evidence)
        unsupported = audit_numbers(briefing, evidence) if source == "claude" else []
        audited += source == "claude"
        flagged += bool(unsupported)
        render(briefing, evidence, source, unsupported)

    if audited:
        print(f"Numeric audit: {audited} model-written briefings checked, "
              f"{flagged} contained a number absent from the evidence packet.")


if __name__ == "__main__":
    main()
