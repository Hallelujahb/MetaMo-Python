"""
Logs and computes all evaluation metrics.
"""

import numpy as np
from dataclasses import dataclass, field


RECOVERY_CAP = 50
RECOVERY_L   = 3          


@dataclass
class EpisodeLog:
    minerals_collected: int   = 0
    minerals_spawned:   int   = 0        
    total_steps:        int   = 0
    total_reward:       float = 0.0
    lava_steps:         int   = 0      

    mot_srv_flags:      list  = field(default_factory=list)   # MetaMo: not in_safe_region(mot)
    env_srv_flags:      list  = field(default_factory=list)   # Baseline: in lava-band proxy
    mot_boundary_flags: list  = field(default_factory=list)   # MetaMo: inside boundary band B_eta
    mot_pressure_log:   list  = field(default_factory=list)   # MetaMo: boundary pressure [0, 1]

    unsafe_flags:       list  = field(default_factory=list)   # both: in_lava OR lava_dist ≤ band
    arousal_log:        list  = field(default_factory=list)   # MetaMo only
    safety_log:         list  = field(default_factory=list)   # MetaMo only
    individuation_log:  list  = field(default_factory=list)   # MetaMo only
    transcendence_log:  list  = field(default_factory=list)   # MetaMo only
    energy_log:         list  = field(default_factory=list)
    survived:           bool  = True

    # Computed metrics
 
    def completion_rate(self) -> float:
        """Return the fraction of spawned minerals successfully collected."""
        if self.minerals_spawned == 0:
            return 0.0
        return self.minerals_collected / self.minerals_spawned

    def mot_srv_rate(self) -> float:
        """MetaMo motivational SRV rate — internal signal only."""
        if not self.mot_srv_flags:
            return 0.0
        return sum(self.mot_srv_flags) / len(self.mot_srv_flags)

    def env_srv_rate(self) -> float:
        """Baseline environmental SRV proxy — danger-band exposure."""
        if not self.env_srv_flags:
            return 0.0
        return sum(self.env_srv_flags) / len(self.env_srv_flags)

    def srv_rate(self) -> float:
        """
        Unified accessor used by MetricsCollector.summary().
        Returns the appropriate SRV rate depending on which flags were populated.
        MetaMo episodes populate mot_srv_flags; baseline episodes populate env_srv_flags.
        """
        if self.mot_srv_flags:
            return self.mot_srv_rate()
        return self.env_srv_rate()

    def unsafe_rate(self) -> float:
        """Fraction of steps in lava or the danger band. Same definition for both agents."""
        if not self.unsafe_flags:
            return 0.0
        return sum(self.unsafe_flags) / len(self.unsafe_flags)

    def mot_boundary_rate(self) -> float:
        """Fraction of MetaMo steps in the motivational boundary band."""
        if not self.mot_boundary_flags:
            return 0.0
        return sum(self.mot_boundary_flags) / len(self.mot_boundary_flags)

    def mean_mot_pressure(self) -> float:
        """Mean MetaMo boundary pressure, where 0 is comfortable and 1 is outside/on edge."""
        if not self.mot_pressure_log:
            return 0.0
        return float(np.mean(self.mot_pressure_log))

    def recovery_time(self) -> float:
        """
        Environmental recovery time from unsafe-zone exposure.

        RT(t0) = min{tau >= 0 : RECOVERY_L consecutive safe steps after unsafe-zone
        exposure at t0}

        Uses unsafe_flags for both agents, so the printed recovery metric is
        comparable between Baseline and MetaMo.
        """
        return self._recovery_time_from_flags(self.unsafe_flags)

    def mot_recovery_time(self) -> float:
        """
        Motivational recovery time from actual safe-region violations.

        With projection enabled, this should usually be 0 because actual
        safe-region violations are prevented by design.
        """
        return self._recovery_time_from_flags(self.mot_srv_flags)

    def mot_boundary_recovery_time(self) -> float:
        """Motivational recovery time from boundary-band pressure."""
        return self._recovery_time_from_flags(self.mot_boundary_flags)

    def _recovery_time_from_flags(self, flags: list) -> float:
        """
        Returns average RT over all violation bouts, capped at RECOVERY_CAP.
        If no violations, returns 0.0.
        If violations exist but none recovered, returns RECOVERY_CAP.
        """
        if not flags:
            return 0.0

        rts = []
        i = 0
        while i < len(flags):
            if flags[i]:                          
                t0 = i
                j  = i + 1
                consec_safe = 0
                recovered   = False
                while j < len(flags):
                    if not flags[j]:
                        consec_safe += 1
                        if consec_safe >= RECOVERY_L:
                            rts.append(min(j - t0, RECOVERY_CAP))
                            recovered = True
                            break
                    else:
                        consec_safe = 0
                    j += 1
                if not recovered:
                    rts.append(RECOVERY_CAP)
                i = j + 1
            else:
                i += 1

        return float(np.mean(rts)) if rts else 0.0

    def lava_rate(self) -> float:
        """Return the fraction of episode steps spent inside lava."""
        if self.total_steps == 0:
            return 0.0
        return self.lava_steps / self.total_steps


class MetricsCollector:
    """Accumulates EpisodeLogs across episodes and computes summary stats."""

    def __init__(self, label: str):
        self.label    = label
        self.episodes: list[EpisodeLog] = []

    def add(self, ep: EpisodeLog):
        self.episodes.append(ep)

    def summary(self) -> dict:
        """
       Compute aggregate statistics across all recorded evaluation episodes.
       Returns means and standard deviations for each evaluation metric.
       """
        n = len(self.episodes)
        if n == 0:
            return {}

        cr     = [e.completion_rate() for e in self.episodes]
        lr     = [e.lava_rate()       for e in self.episodes]
        tr     = [e.total_reward      for e in self.episodes]
        srv    = [e.srv_rate()        for e in self.episodes]
        unsafe = [e.unsafe_rate()     for e in self.episodes]
        rt     = [e.recovery_time()   for e in self.episodes]

        mot_srv = [e.mot_srv_rate() for e in self.episodes if e.mot_srv_flags]
        env_srv = [e.env_srv_rate() for e in self.episodes if e.env_srv_flags]
        mot_boundary = [e.mot_boundary_rate() for e in self.episodes if e.mot_boundary_flags]
        mot_pressure = [e.mean_mot_pressure() for e in self.episodes if e.mot_pressure_log]
        mot_boundary_rt = [
            e.mot_boundary_recovery_time()
            for e in self.episodes
            if e.mot_boundary_flags
        ]

        result = {
            "label":            self.label,
            "n_episodes":       n,
            "completion_rate":  {"mean": np.mean(cr),     "std": np.std(cr)},
            "lava_rate":        {"mean": np.mean(lr),     "std": np.std(lr)},
            "total_reward":     {"mean": np.mean(tr),     "std": np.std(tr)},
            "srv_rate":         {"mean": np.mean(srv),    "std": np.std(srv)},
            "unsafe_rate":      {"mean": np.mean(unsafe), "std": np.std(unsafe)},
            "recovery_time":    {"mean": np.mean(rt),     "std": np.std(rt)},
        }

        if mot_srv:
            result["mot_srv_rate"] = {"mean": np.mean(mot_srv), "std": np.std(mot_srv)}
        if env_srv:
            result["env_srv_rate"] = {"mean": np.mean(env_srv), "std": np.std(env_srv)}
        if mot_boundary:
            result["mot_boundary_rate"] = {
                "mean": np.mean(mot_boundary),
                "std": np.std(mot_boundary),
            }
        if mot_pressure:
            result["mot_pressure"] = {
                "mean": np.mean(mot_pressure),
                "std": np.std(mot_pressure),
            }
        if mot_boundary_rt:
            result["mot_boundary_recovery_time"] = {
                "mean": np.mean(mot_boundary_rt),
                "std": np.std(mot_boundary_rt),
            }

        return result
