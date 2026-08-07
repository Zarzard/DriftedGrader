"""C8 environment: 10x10 gridworld (spec v0.2 Section 3, C8).

Frozen apparatus: the map layout is deterministic (fixed internal seed), shared by
every run. Actions {up, down, left, right, stay}, slip 0.05 (executed action is
uniformly one of the OTHER four with that probability), episodes are H=12-step
segments from uniform random spawn, no termination (goal is per-step and
non-absorbing -- amendment A6).

Step features phi(s, a_exec, s') in R^8:
    [0] on_goal_cell(s')        (per-step, non-absorbing)
    [1] -manhattan(s', goal)/18 (goal proximity)
    [2] lava(s')  [3] water(s')  [4] grass(s')
    [5] wall_bump (executed move was blocked)
    [6] stay_action (executed action was 'stay')
    [7] 1 (constant)
Cell types partition the grid, so phi0+phi2+phi3+phi4 = 1 per step and phi7 = 1:
segment gaps annihilate {e7, e0+e2+e3+e4} -> achievable-gap rank = 6 (test 10).

Exact machinery for policy work: transition matrices P[a] (S x S) and expected
per-step features F[s, a] in R^8, both including slip, built once.
"""
from __future__ import annotations

import numpy as np

N = 10
S = N * N
A = 5                     # up, down, left, right, stay
STAY = 4
H = 12                    # segment length
SLIP = 0.05
GAMMA = 0.97
D_PHI = 8
_MAP_SEED = 20260805      # frozen map

# feature indices
F_GOAL, F_PROX, F_LAVA, F_WATER, F_GRASS, F_BUMP, F_STAY, F_CONST = range(8)

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


class GridWorld:
    """Deterministic-layout gridworld with exact MDP matrices."""

    def __init__(self) -> None:
        rng = np.random.default_rng(_MAP_SEED)
        self.goal = S - 1  # cell (9, 9)
        others = np.array([s for s in range(S) if s != self.goal])
        picks = rng.choice(others, size=14, replace=False)
        self.lava = set(picks[:6].tolist())
        self.water = set(picks[6:].tolist())

        # deterministic step for each EXECUTED action: next state + wall bump flag
        nxt = np.empty((S, A), dtype=int)
        bump = np.zeros((S, A), dtype=bool)
        for s in range(S):
            r, c = divmod(s, N)
            for a in range(A):
                dr, dc = _MOVES[a]
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    nxt[s, a] = rr * N + cc
                else:
                    nxt[s, a] = s
                    bump[s, a] = True
        self._nxt, self._bump = nxt, bump

        # per-(s, a_exec) step features (transitions are deterministic given a_exec)
        phi_exec = np.zeros((S, A, D_PHI))
        for s in range(S):
            for a in range(A):
                phi_exec[s, a] = self._phi_step(s, a)
        self._phi_exec = phi_exec

        # slip mixing: P(a_exec | a_chosen)
        mix = np.full((A, A), SLIP / (A - 1))
        np.fill_diagonal(mix, 1.0 - SLIP)
        self._mix = mix

        # exact MDP matrices under chosen actions
        self.P = np.zeros((A, S, S))       # P[a, s, s']
        self.F = np.zeros((S, A, D_PHI))   # expected step features
        for a in range(A):
            for s in range(S):
                for ae in range(A):
                    self.P[a, s, nxt[s, ae]] += mix[a, ae]
                self.F[s, a] = mix[a] @ phi_exec[s]

    def _phi_step(self, s: int, a_exec: int) -> np.ndarray:
        s2 = self._nxt[s, a_exec]
        r2, c2 = divmod(s2, N)
        gr, gc = divmod(self.goal, N)
        phi = np.zeros(D_PHI)
        phi[F_GOAL] = 1.0 if s2 == self.goal else 0.0
        phi[F_PROX] = -(abs(r2 - gr) + abs(c2 - gc)) / (2 * (N - 1))
        phi[F_LAVA] = 1.0 if s2 in self.lava else 0.0
        phi[F_WATER] = 1.0 if s2 in self.water else 0.0
        phi[F_GRASS] = 1.0 - phi[F_GOAL] - phi[F_LAVA] - phi[F_WATER]
        phi[F_BUMP] = 1.0 if self._bump[s, a_exec] else 0.0
        phi[F_STAY] = 1.0 if a_exec == STAY else 0.0
        phi[F_CONST] = 1.0
        return phi

    # ------------------------------------------------------------------ #
    def rollout_segment(self, policy: np.ndarray, rng: np.random.Generator,
                        start: int | None = None) -> np.ndarray:
        """One H-step segment; returns summed features (R^8). ``policy`` is (S, A)."""
        s = int(rng.integers(S)) if start is None else start
        total = np.zeros(D_PHI)
        for _ in range(H):
            a = int(rng.choice(A, p=policy[s]))
            ae = a if rng.random() >= SLIP else int(rng.choice([x for x in range(A) if x != a]))
            total += self._phi_exec[s, ae]
            s = int(self._nxt[s, ae])
        return total


_ENV: GridWorld | None = None


def get_env() -> GridWorld:
    """The frozen shared instance."""
    global _ENV
    if _ENV is None:
        _ENV = GridWorld()
    return _ENV


def uniform_policy() -> np.ndarray:
    return np.full((S, A), 1.0 / A)


TRUE_W = np.array([+1.0, +0.3, -1.0, -0.3, 0.0, -0.1, -0.02, -0.02])
