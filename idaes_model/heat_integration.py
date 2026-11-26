"""
Generic pinch analysis (problem-table algorithm) for any flowsheet.

Feed it a list of process streams; it returns the minimum hot and cold
utility, the pinch temperature, the grand composite curve, and how much
of the cold utility has to be rejected below the cooling-water
temperature (i.e. needs a chiller). Nothing here is specific to the
ammonia plant, so the urea / DEF streams can be appended later for a
plant-wide study.

Conventions
-----------
* duty_kw > 0 always; the stream type ("hot" = needs cooling, "cold" =
  needs heating) comes from t_in > t_out or t_in < t_out.
* Constant-temperature duties (condensers, reboilers, isothermal
  reactors) are given as t_in, t_out differing by 0.1 K.
* Each stream may carry its own dT_min contribution (half of the
  approach temperature, Linnhoff's "individual dT_min/2").

Example
-------
>>> from heat_integration import Stream, pinch
>>> r = pinch([Stream("reactor", 60.1, 60.0, 200.0),
...            Stream("feed", 10.0, 100.0, 150.0)], dt_min=10.0)
>>> round(r.q_hot_min, 1), round(r.q_cold_min, 1)
"""

from dataclasses import dataclass, field


@dataclass
class Stream:
    name: str
    t_in: float      # degC
    t_out: float     # degC
    duty_kw: float   # > 0
    dt_contrib: float = None   # default: dt_min / 2
    section: str = ""          # free label, e.g. "ammonia", "urea"

    @property
    def kind(self):
        return "hot" if self.t_in > self.t_out else "cold"

    @property
    def cp(self):
        """Heat-capacity flowrate [kW/K]."""
        return self.duty_kw / abs(self.t_in - self.t_out)


@dataclass
class PinchResult:
    q_hot_min: float
    q_cold_min: float
    t_pinch_shifted: float
    q_hot_no_hi: float      # hot utility with no heat integration
    q_cold_no_hi: float
    gcc: list = field(default_factory=list)   # [(T_shifted, heat_flow)]
    dt_min: float = 10.0

    @property
    def recovered(self):
        return self.q_hot_no_hi - self.q_hot_min

    def cold_below(self, t_real):
        """Cold utility that must be removed below real temperature
        t_real (e.g. 35 C, the lowest a cooling-water cooler can reach).

        The GCC value at a shifted temperature T is the heat cascading
        down across T. Below the pinch that is surplus released ABOVE T,
        which a utility at T can take; what must go below T is the surplus
        released below it: q_cold_min - min(GCC at or below T).
        (Before 2026-11-26 this returned GCC(T) itself - the wrong side.)"""
        ts = t_real - self.dt_min / 2.0
        if ts >= self.t_pinch_shifted:
            return self.q_cold_min
        lows = [q for t, q in self.gcc if t <= ts] + [_interp(self.gcc, ts)]
        return max(self.q_cold_min - min(lows), 0.0)

    def hot_above(self, t_real):
        """Hot utility that must be supplied above real temperature t_real:
        q_hot_min - min(GCC at or above T) (same correction as
        cold_below)."""
        ts = t_real + self.dt_min / 2.0
        if ts <= self.t_pinch_shifted:
            return self.q_hot_min
        highs = [q for t, q in self.gcc if t >= ts] + [_interp(self.gcc, ts)]
        return max(self.q_hot_min - min(highs), 0.0)

    def summary(self):
        return (f"dT_min = {self.dt_min:.0f} K | no integration: hot "
                f"{self.q_hot_no_hi:.1f} kW, cold {self.q_cold_no_hi:.1f} kW"
                f" | targets: hot {self.q_hot_min:.1f} kW, cold "
                f"{self.q_cold_min:.1f} kW | recovered {self.recovered:.1f} "
                f"kW | pinch (shifted) {self.t_pinch_shifted:.1f} C")


def _interp(gcc, t):
    pts = sorted(gcc)
    for (t0, q0), (t1, q1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return q0 if t1 == t0 else q0 + (q1 - q0) * (t - t0) / (t1 - t0)
    return pts[0][1] if t < pts[0][0] else pts[-1][1]


def pinch(streams, dt_min=10.0):
    """Problem-table algorithm. Returns a PinchResult."""
    shifted = []
    for s in streams:
        d = s.dt_contrib if s.dt_contrib is not None else dt_min / 2.0
        sh = -d if s.kind == "hot" else d
        shifted.append((s, s.t_in + sh, s.t_out + sh))
    temps = sorted({t for _, a, b in shifted for t in (a, b)}, reverse=True)

    # heat surplus of each temperature interval (top -> bottom)
    surplus = []
    for hi, lo in zip(temps, temps[1:]):
        net = 0.0
        for s, a, b in shifted:
            top, bot = max(a, b), min(a, b)
            if top >= hi and bot <= lo:
                net += s.cp * (hi - lo) * (1 if s.kind == "hot" else -1)
        surplus.append(net)

    cascade = [0.0]
    for q in surplus:
        cascade.append(cascade[-1] + q)
    q_hot = max(0.0, -min(cascade))
    flows = [q_hot + c for c in cascade]
    q_cold = flows[-1]
    i_pinch = min(range(len(flows)), key=lambda i: flows[i])
    gcc = list(zip(temps, flows))

    hot_total = sum(s.duty_kw for s in streams if s.kind == "cold")
    cold_total = sum(s.duty_kw for s in streams if s.kind == "hot")
    return PinchResult(q_hot_min=q_hot, q_cold_min=q_cold,
                       t_pinch_shifted=temps[i_pinch],
                       q_hot_no_hi=hot_total, q_cold_no_hi=cold_total,
                       gcc=gcc, dt_min=dt_min)


def plot_gcc(result, path, title="Grand composite curve"):
    """Save the grand composite curve (needs matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, q = zip(*result.gcc)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(q, t, color="#2a6fb5", lw=2)
    ax.axhline(result.t_pinch_shifted, color="#999999", ls="--", lw=1)
    ax.set_xlabel("Net heat flow [kW]")
    ax.set_ylabel("Shifted temperature [°C]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
