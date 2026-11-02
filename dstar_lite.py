"""D* Lite (Koenig & Likhachev, 2002) on an 8-connected occupancy grid.

Searches backward from the goal cells to the robot, so when the robot
moves and some cells' occupancy changes, only the part of the search those
changes actually invalidate is repaired -- rather than replanning from
scratch at every sensing cycle. Multiple goal cells (a whole finish line)
are supported by fixing every goal's rhs at 0, equivalent to a single
virtual goal joined to each at zero cost. Besides hard blocks, cells can
carry a soft penalty (a multiplier on the cost of stepping through them),
which D* Lite repairs around exactly like a block appearing or clearing.

Cells are (i, j) integer pairs; the caller owns the mapping to and from
world coordinates (live_planning.py)."""

import heapq
import math

INF = math.inf
SQRT2 = math.sqrt(2)
STEPS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


class DStarLite:
    def __init__(self, nx, ny, goals, start, blocked=(), penalty=None):
        self.nx, self.ny = nx, ny
        self.goals = set(goals)
        self.blocked = set(blocked)
        self.penalty = dict(penalty or {})
        self.g, self.rhs = {}, {}
        self.queue = []      # heap of (key, cell); entries whose key no longer
        self.queued = {}     # matches queued[cell] are stale and skipped
        self.km = 0.0
        self.start = self.last = start
        self.expansions = 0  # cumulative, for reporting how much each replan cost
        for s in self.goals:
            self.rhs[s] = 0.0
            self._push(s)
        self._compute()

    # --- graph ---------------------------------------------------------------
    def _neighbors(self, s):
        i, j = s
        for di, dj in STEPS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.nx and 0 <= nj < self.ny:
                yield ni, nj

    def cost(self, a, b):
        """Infinite if either end is blocked, or if a diagonal step would cut
        the corner of a blocked cell; otherwise the step's length in cells,
        scaled up by the two ends' average penalty. Never below the bare
        length, so the octile heuristic stays consistent."""
        if a in self.blocked or b in self.blocked:
            return INF
        di, dj = b[0] - a[0], b[1] - a[1]
        length = 1.0
        if di and dj:
            if (a[0] + di, a[1]) in self.blocked or (a[0], a[1] + dj) in self.blocked:
                return INF
            length = SQRT2
        pen = self.penalty
        if pen:
            return length * (1.0 + 0.5 * (pen.get(a, 0.0) + pen.get(b, 0.0)))
        return length

    @staticmethod
    def h(a, b):
        """Octile distance: consistent for 8-connected unit/sqrt2 steps."""
        di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(di, dj) + (SQRT2 - 1) * min(di, dj)

    # --- D* Lite core --------------------------------------------------------
    def _key(self, s):
        # Rounded so keys that tie in exact arithmetic tie in floating point
        # too: sums of 1s and sqrt2s reached by different routes can differ
        # in the last bit, and a top key a hair above the start's stops the
        # search one expansion early, leaving stale g-values that path()
        # then follows into a longer-than-optimal route.
        m = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (round(m + self.h(self.start, s) + self.km, 9), round(m, 9))

    def _push(self, s):
        k = self._key(s)
        self.queued[s] = k
        heapq.heappush(self.queue, (k, s))

    def _update_vertex(self, u):
        if u not in self.goals:
            self.rhs[u] = min((self.cost(u, s) + self.g.get(s, INF) for s in self._neighbors(u)), default=INF)
        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self._push(u)
        else:
            self.queued.pop(u, None)

    def _compute(self):
        while self.queue:
            k_old, u = self.queue[0]
            if self.queued.get(u) != k_old:
                heapq.heappop(self.queue)
                continue
            start_consistent = self.g.get(self.start, INF) == self.rhs.get(self.start, INF)
            if not k_old < self._key(self.start) and start_consistent:
                break
            heapq.heappop(self.queue)
            del self.queued[u]
            self.expansions += 1
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u)
            elif self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for s in self._neighbors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for s in self._neighbors(u):
                    self._update_vertex(s)

    # --- public --------------------------------------------------------------
    def update(self, start, blocked, penalty=None):
        """Moves the robot to `start` and replaces the blocked set and
        penalties, repairing only around the cells whose occupancy or
        penalty actually changed."""
        blocked = set(blocked)
        penalty = dict(penalty or {})
        changed = self.blocked ^ blocked
        changed.update(c for c in self.penalty.keys() | penalty.keys()
                       if self.penalty.get(c, 0.0) != penalty.get(c, 0.0))
        self.penalty = penalty
        if start != self.start:
            self.km += self.h(self.last, start)
            self.last = self.start = start
        self.blocked = blocked
        for c in changed:
            self._update_vertex(c)
            for s in self._neighbors(c):
                self._update_vertex(s)
        self._compute()

    def path(self, max_cells=200):
        """Cells from the start toward the goal, following the cheapest
        cost-plus-g successor; None if the goal is currently unreachable."""
        if self.g.get(self.start, INF) == INF and self.rhs.get(self.start, INF) == INF:
            return None
        s, out = self.start, [self.start]
        while s not in self.goals and len(out) < max_cells:
            nxt = min(self._neighbors(s), key=lambda n: self.cost(s, n) + self.g.get(n, INF))
            if self.cost(s, nxt) + self.g.get(nxt, INF) == INF:
                return None
            s = nxt
            out.append(s)
        return out
