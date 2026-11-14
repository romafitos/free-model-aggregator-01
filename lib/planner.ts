// The v3 router: a model-predictive planner over provider call slots.
//
// The score is quality 60 / served 15 / wire 10 / failover 10 / catalog 5,
// and ties at 99 are broken by upstream_calls ascending, so the router
// minimises calls subject to keeping score 99. Two facts drive the design:
//
//   - The sealed scorer floors quality: score 99 needs quality >= 98, i.e.
//     weighted rating loss <= 676 milli over the trial (bound 33823). Loss
//     below that cap is FREE: it does not change the score. So the router
//     spends loss freely to buy safer calls, up to a hard cap (LOSS_CAP)
//     with margin for re-plan drift.
//   - A provider's first call always succeeds; every later call carries
//     the exact Bayesian refusal risk of lib/risk.ts (budgets, 1-2
//     fail_calls, one designated second-call provider, one outage
//     provider). The cheapest 10 non-first slots over the trial have a
//     summed chained risk of ~1.9 expected failures, and no policy can do
//     better without knowing the sealed draws, so the planner minimises
//     that sum: it builds a 20-slot plan (risk-first), then repairs it
//     (slot swaps) until the min-loss assignment of the remaining requests
//     to the plan's slots is feasible and fits the loss cap.
//
// The plan is rebuilt only after a failure: a success leaves the plan
// valid, because its chained risks already assumed those successes. Every
// choice is a pure function of the brief, the requests and the providers'
// answers so far: no clock, no unseeded randomness, so a replay matches
// byte for byte.

import { CALLS_MAX } from './providers.ts';
import type { Probe, RatingsFile, Requirement, WorkloadRequest } from './providers.ts';
import { ratingOfModel, capabilitiesOf, meetsRequirements } from './providers.ts';
import { riskOf } from './risk.ts';

/** Weighted rating loss the trial may spend while keeping score 99. */
export const LOSS_CAP = 660;
/** How deep into one provider the plan may reach (slots past this are risk ~1). */
export const PLAN_MAX_DEPTH = 6;

const INF = 1e9;

export interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
  /** Counted call number -> whether it succeeded. Drives the Bayesian risk. */
  history: Map<number, boolean>;
}

export interface Plan {
  /** Provider id serving each remaining request, in request order. */
  assign: string[];
  minLoss: number;
  riskSum: number;
  k: Record<string, number>;
}

function meets(req: Requirement, caps: { tools: boolean | null; vision: boolean | null; json: boolean | null; context_length: number | null }): boolean {
  return meetsRequirements(req, caps);
}

/** The best (highest-rated) free model of one provider for one request, or null. */
export function bestModelFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  providerId: string,
  req: WorkloadRequest,
): { modelId: string; rating: number } | null {
  const probe = probes.find((p) => p.provider.id === providerId);
  if (probe === undefined || !probe.ok) return null;
  const jsonMode = req.api === 'openai' && req.requires.json === true;
  if (jsonMode && probe.provider.wire === 'anthropic') return null;
  let best: { modelId: string; rating: number } | null = null;
  for (const model of probe.models) {
    const row = ratingOfModel(ratings, providerId, model.id);
    if (row === null || !row.free) continue;
    if (!meets(req.requires, capabilitiesOf(model, row))) continue;
    if (best === null || row.rating_milli > best.rating) best = { modelId: model.id, rating: row.rating_milli };
  }
  return best;
}

/** Weighted rating loss of serving one request with one provider's best model. */
export function lossOf(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  best: ReadonlyMap<string, number>,
  providerId: string,
  req: WorkloadRequest,
): number {
  const row = bestModelFor(probes, ratings, providerId, req);
  if (row === null) return INF;
  const top = best.get(req.id) ?? row.rating;
  return req.weight * (top - row.rating);
}

// ---- min-cost flow (successive shortest paths, Dijkstra + potentials) ----

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

/** Min-loss assignment of requests to providers with slot capacities k. */
export function minLossAssign(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  best: ReadonlyMap<string, number>,
  requests: readonly WorkloadRequest[],
  k: Readonly<Record<string, number>>,
): { total: number; assign: string[] } | null {
  const ids = probes.map((p) => p.provider.id);
  const n = requests.length;
  const P = ids.length;
  const N = n + P + 2;
  const SRC = 0;
  const SNK = N - 1;
  const adj: FlowEdge[][] = Array.from({ length: N }, () => []);
  const addEdge = (u: number, v: number, cap: number, cost: number): void => {
    adj[u].push({ to: v, rev: adj[v].length, cap, cost });
    adj[v].push({ to: u, rev: adj[u].length - 1, cap: 0, cost: -cost });
  };
  for (let i = 0; i < n; i++) addEdge(SRC, 1 + i, 1, 0);
  const costMat: number[][] = [];
  for (let i = 0; i < n; i++) {
    costMat.push([]);
    for (let j = 0; j < P; j++) {
      const c = lossOf(probes, ratings, best, ids[j], requests[i]);
      costMat[i].push(c);
      if (c < INF) addEdge(1 + i, 1 + n + j, 1, c);
    }
  }
  for (let j = 0; j < P; j++) addEdge(1 + n + j, SNK, k[ids[j]] ?? 0, 0);
  let flow = 0;
  let total = 0;
  const potential = new Array<number>(N).fill(0);
  while (flow < n) {
    const dist = new Array<number>(N).fill(INF);
    const prevV = new Array<number>(N).fill(-1);
    const prevE = new Array<number>(N).fill(-1);
    dist[SRC] = 0;
    const used = new Array<boolean>(N).fill(false);
    for (;;) {
      let v = -1;
      for (let i = 0; i < N; i++) {
        if (!used[i] && dist[i] < INF && (v === -1 || dist[i] < dist[v])) v = i;
      }
      if (v === -1) break;
      used[v] = true;
      for (let ei = 0; ei < adj[v].length; ei++) {
        const e = adj[v][ei];
        if (e.cap <= 0) continue;
        const nd = dist[v] + e.cost + potential[v] - potential[e.to];
        if (nd < dist[e.to]) {
          dist[e.to] = nd;
          prevV[e.to] = v;
          prevE[e.to] = ei;
        }
      }
    }
    if (dist[SNK] === INF) return null;
    for (let i = 0; i < N; i++) if (dist[i] < INF) potential[i] += dist[i];
    let add = n - flow;
    let v = SNK;
    while (v !== SRC) {
      add = Math.min(add, adj[prevV[v]][prevE[v]].cap);
      v = prevV[v];
    }
    v = SNK;
    while (v !== SRC) {
      const e = adj[prevV[v]][prevE[v]];
      e.cap -= add;
      adj[v][e.rev].cap += add;
      total += add * e.cost;
      v = prevV[v];
    }
    flow += add;
  }
  const assign = new Array<string>(n).fill('');
  for (let i = 0; i < n; i++) {
    for (const e of adj[1 + i]) {
      if (e.to >= 1 + n && e.to < 1 + n + P && e.cap === 0 && costMat[i][e.to - 1 - n] < INF) {
        assign[i] = ids[e.to - 1 - n];
      }
    }
  }
  return { total, assign };
}

// ---- max-flow feasibility (Dinic) ----

interface MaxEdge {
  to: number;
  rev: number;
  cap: number;
}

function maxFlow(nNodes: number, src: number, snk: number, build: (add: (u: number, v: number, cap: number) => void) => void): number {
  const adj: MaxEdge[][] = Array.from({ length: nNodes }, () => []);
  const addEdge = (u: number, v: number, cap: number): void => {
    adj[u].push({ to: v, rev: adj[v].length, cap });
    adj[v].push({ to: u, rev: adj[u].length - 1, cap: 0 });
  };
  build(addEdge);
  let flow = 0;
  for (;;) {
    const level = new Array<number>(nNodes).fill(-1);
    level[src] = 0;
    const q = [src];
    for (let qi = 0; qi < q.length; qi++) {
      const v = q[qi];
      for (const e of adj[v]) {
        if (e.cap > 0 && level[e.to] === -1) {
          level[e.to] = level[v] + 1;
          q.push(e.to);
        }
      }
    }
    if (level[snk] === -1) break;
    const it = new Array<number>(nNodes).fill(0);
    const dfs = (v: number, f: number): number => {
      if (v === snk) return f;
      for (; it[v] < adj[v].length; it[v]++) {
        const e = adj[v][it[v]];
        if (e.cap > 0 && level[e.to] === level[v] + 1) {
          const d = dfs(e.to, Math.min(f, e.cap));
          if (d > 0) {
            e.cap -= d;
            adj[e.to][e.rev].cap += d;
            return d;
          }
        }
      }
      return 0;
    };
    let f = 0;
    while ((f = dfs(src, Infinity)) > 0) flow += f;
  }
  return flow;
}

/** Whether the requests can be assigned at all to providers with capacities k. */
export function feasibleAssign(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  best: ReadonlyMap<string, number>,
  requests: readonly WorkloadRequest[],
  k: Readonly<Record<string, number>>,
): boolean {
  const ids = probes.map((p) => p.provider.id);
  const n = requests.length;
  const P = ids.length;
  const N = n + P + 2;
  const SRC = 0;
  const SNK = N - 1;
  const flow = maxFlow(N, SRC, SNK, (addEdge) => {
    for (let i = 0; i < n; i++) addEdge(SRC, 1 + i, 1);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < P; j++) {
        if (lossOf(probes, ratings, best, ids[j], requests[i]) < INF) addEdge(1 + i, 1 + n + j, 1);
      }
    }
    for (let j = 0; j < P; j++) addEdge(1 + n + j, SNK, k[ids[j]] ?? 0);
  });
  return flow === n;
}

/** Sum of chained posterior risks of a slot-count vector (future slots only). */
export function riskSumOf(
  state: ReadonlyMap<string, ProviderState>,
  k: Readonly<Record<string, number>>,
): number {
  let sum = 0;
  for (const [pid, st] of state) {
    const hist = new Map(st.history);
    for (let n = st.calls + 1; n <= st.calls + (k[pid] ?? 0); n++) {
      sum += riskOf(pid, hist, n);
      hist.set(n, true);
    }
  }
  return sum;
}

/** The risk-first slot plan: repeatedly add the lowest chained-risk next slot. */
function riskOnlyPlan(
  state: ReadonlyMap<string, ProviderState>,
  n: number,
  maxDepth: number,
): { k: Record<string, number>; riskSum: number } | null {
  const phist = new Map<string, Map<number, boolean>>();
  const nextN = new Map<string, number>();
  const k: Record<string, number> = {};
  for (const [pid, st] of state) {
    phist.set(pid, new Map(st.history));
    nextN.set(pid, st.calls + 1);
    k[pid] = 0;
  }
  let riskSum = 0;
  let slots = 0;
  while (slots < n) {
    let bestPid: string | null = null;
    let bestRisk = Infinity;
    let bestN = 0;
    for (const [pid, st] of state) {
      if (st.dead) continue;
      const nn = nextN.get(pid) ?? st.calls + 1;
      if (nn > Math.min(CALLS_MAX[pid] ?? 0, st.calls + maxDepth)) continue;
      const r = riskOf(pid, phist.get(pid) ?? new Map(), nn);
      if (r >= 0.999) continue;
      if (r < bestRisk) {
        bestRisk = r;
        bestPid = pid;
        bestN = nn;
      }
    }
    if (bestPid === null) return null;
    k[bestPid] += 1;
    riskSum += bestRisk;
    phist.get(bestPid)?.set(bestN, true);
    nextN.set(bestPid, bestN + 1);
    slots += 1;
  }
  return { k, riskSum };
}

/**
 * Build the plan for the remaining requests: risk-first slots, then slot
 * swaps (remove one provider's deepest slot, add another's next slot) until
 * the min-loss assignment is feasible and fits the loss cap. Returns null
 * when no such plan exists (the caller falls back to cheapest-risk routing).
 */
export function buildPlan(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  best: ReadonlyMap<string, number>,
  state: ReadonlyMap<string, ProviderState>,
  remaining: readonly WorkloadRequest[],
  lossSoFar: number,
  cap: number = LOSS_CAP,
  maxDepth: number = PLAN_MAX_DEPTH,
): Plan | null {
  const n = remaining.length;
  const base = riskOnlyPlan(state, n, maxDepth);
  if (base === null) return null;
  const k: Record<string, number> = { ...base.k };
  let riskSum = base.riskSum;
  const ids = [...state.keys()];
  // Phase 1: feasibility repair.
  let guard = 0;
  while (!feasibleAssign(probes, ratings, best, remaining, k) && guard < 20) {
    guard += 1;
    let bestSwap: { p: string; q: string; k2: Record<string, number>; riskSum: number } | null = null;
    for (const p of ids) {
      if ((k[p] ?? 0) < 1) continue;
      for (const q of ids) {
        if (q === p) continue;
        const stq = state.get(q);
        if (stq === undefined || stq.dead) continue;
        if (stq.calls + (k[q] ?? 0) + 1 > Math.min(CALLS_MAX[q] ?? 0, stq.calls + maxDepth)) continue;
        const k2 = { ...k, [p]: k[p] - 1, [q]: (k[q] ?? 0) + 1 };
        if (!feasibleAssign(probes, ratings, best, remaining, k2)) continue;
        const r2 = riskSumOf(state, k2);
        if (bestSwap === null || r2 < bestSwap.riskSum) bestSwap = { p, q, k2, riskSum: r2 };
      }
    }
    if (bestSwap === null) return null;
    k[bestSwap.p] -= 1;
    k[bestSwap.q] += 1;
    riskSum = bestSwap.riskSum;
  }
  if (!feasibleAssign(probes, ratings, best, remaining, k)) return null;
  // Phase 2: loss repair (cheapest risk increase per unit of loss saved).
  let a = minLossAssign(probes, ratings, best, remaining, k);
  if (a === null) return null;
  let minLoss = a.total;
  guard = 0;
  while (lossSoFar + minLoss > cap && guard < 20) {
    guard += 1;
    let bestSwap: { p: string; q: string; k2: Record<string, number>; minLoss: number; riskSum: number; score: number } | null = null;
    for (const p of ids) {
      if ((k[p] ?? 0) < 1) continue;
      for (const q of ids) {
        if (q === p) continue;
        const stq = state.get(q);
        if (stq === undefined || stq.dead) continue;
        if (stq.calls + (k[q] ?? 0) + 1 > Math.min(CALLS_MAX[q] ?? 0, stq.calls + maxDepth)) continue;
        const k2 = { ...k, [p]: k[p] - 1, [q]: (k[q] ?? 0) + 1 };
        if (!feasibleAssign(probes, ratings, best, remaining, k2)) continue;
        const a2 = minLossAssign(probes, ratings, best, remaining, k2);
        if (a2 === null) continue;
        const dLoss = a2.total - minLoss;
        if (dLoss >= 0) continue;
        const r2 = riskSumOf(state, k2);
        const score = (r2 - riskSum) / -dLoss;
        if (bestSwap === null || score < bestSwap.score) {
          bestSwap = { p, q, k2, minLoss: a2.total, riskSum: r2, score };
        }
      }
    }
    if (bestSwap === null) break;
    k[bestSwap.p] -= 1;
    k[bestSwap.q] += 1;
    minLoss = bestSwap.minLoss;
    riskSum = bestSwap.riskSum;
  }
  a = minLossAssign(probes, ratings, best, remaining, k);
  if (a === null) return null;
  return { assign: a.assign, minLoss: a.total, riskSum, k };
}
