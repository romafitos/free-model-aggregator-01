// The Bayesian per-call failure risk of a provider.
//
// The generator draws each trial, per provider, from a published prior the
// entry may use (the kit's generator source, the same place CALLS_MAX comes
// from); the per-trial draws themselves are never known:
//
//   budget B ~ uniform over [CALLS_MIN, CALLS_MAX];
//   fail count c ~ uniform over {1, 2};
//   fail_calls ~ a uniform c-subset of [2, B-1] (a provider's first call
//     always succeeds: fail_calls and outages start at 2);
//   one provider per trial is the designated second-call provider (each
//     provider with probability 1/10): its fail_calls are forced to hold 2;
//   one provider per trial has an outage (each with probability 1/10):
//     outage_from ~ uniform over [2, B-1], and every call at or past it
//     refuses.
//
// riskOf(providerId, history, nextN) is the exact posterior probability that
// the provider's next call (its nextN-th counted attempt) refuses, given the
// observed history of that provider's counted attempts so far (a map from
// call number to whether it succeeded). Worlds inconsistent with the history
// are dropped; the rest are weighted by the prior. Two facts the posterior
// captures that a fixed table cannot:
//
//   - a failure spends one fail_call, so the next number is usually safer
//     (retrying after a one-off refusal is cheap);
//   - a run of successes pins the fail_calls down: if calls 1..10 all
//     succeeded, the only consistent worlds put the remaining fail_call at
//     11, so call 11 is certainly refused.
//
// Deterministic: a pure function of the history. No clock, no randomness, so
// a trial replays byte for byte. Results are cached per (provider, history,
// nextN); histories only grow when that provider is attempted.

import { CALLS_MAX, CALLS_MIN } from './providers.ts';

interface World {
  budget: number;
  fails: Set<number>;
  outageFrom: number | null;
  weight: number;
}

const worldsCache = new Map<string, World[]>();

function subsetsOf(arr: readonly number[], k: number): number[][] {
  const out: number[][] = [];
  const rec = (start: number, picked: number[]): void => {
    if (picked.length === k) {
      out.push([...picked]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      picked.push(arr[i]);
      rec(i + 1, picked);
      picked.pop();
    }
  };
  rec(0, []);
  return out;
}

/** Every prior world for one provider, each once, weighted by the prior. */
function worldsFor(providerId: string): World[] {
  const cached = worldsCache.get(providerId);
  if (cached !== undefined) return cached;
  const lo = CALLS_MIN[providerId];
  const hi = CALLS_MAX[providerId];
  const worlds: World[] = [];
  for (let budget = lo; budget <= hi; budget++) {
    const nums: number[] = [];
    for (let n = 2; n <= budget - 1; n++) nums.push(n);
    const hiNums = nums.filter((n) => n >= 3);
    for (let c = 1; c <= 2; c++) {
      for (const designated of [false, true]) {
        const pDesignated = designated ? 0.1 : 0.9;
        // The designated provider's fail_calls hold 2 (the generator forces
        // it in, replacing the smallest drawn number); otherwise the set is
        // a uniform c-subset. The uniform-over-higher-numbers approximation
        // for the designated second member matches the simulator the policy
        // was tuned against; the difference is inside the 0.1 prior mass.
        const failsets = designated
          ? subsetsOf(hiNums, c - 1).map((s) => [2, ...s])
          : subsetsOf(nums, c);
        const pSet = 1 / Math.max(1, failsets.length);
        for (const failset of failsets) {
          for (const outageFrom of [null, ...nums]) {
            const pOutage = outageFrom === null ? 0.9 : 0.1 / nums.length;
            const weight = (1 / (hi - lo + 1)) * 0.5 * pSet * pDesignated * pOutage;
            worlds.push({ budget, fails: new Set(failset), outageFrom, weight });
          }
        }
      }
    }
  }
  worldsCache.set(providerId, worlds);
  return worlds;
}

function callFails(world: World, n: number): boolean {
  if (n > world.budget) return true;
  if (world.outageFrom !== null && n >= world.outageFrom) return true;
  if (world.fails.has(n)) return true;
  return false;
}

const riskCache = new Map<string, number>();

/**
 * P(the provider's nextN-th counted call refuses | history). history maps a
 * counted call number to whether that call succeeded. A provider whose
 * history is empty and whose next call is its first has risk 0.
 */
export function riskOf(providerId: string, history: ReadonlyMap<number, boolean>, nextN: number): number {
  const signature = `${providerId}|${nextN}|${[...history.entries()].map(([n, ok]) => `${n}${ok ? '1' : '0'}`).join(',')}`;
  const cached = riskCache.get(signature);
  if (cached !== undefined) return cached;
  let total = 0;
  let bad = 0;
  for (const world of worldsFor(providerId)) {
    let consistent = true;
    for (const [n, succeeded] of history) {
      if (callFails(world, n) === succeeded) {
        consistent = false;
        break;
      }
    }
    if (!consistent) continue;
    total += world.weight;
    if (callFails(world, nextN)) bad += world.weight;
  }
  const risk = total === 0 ? 1 : bad / total;
  riskCache.set(signature, risk);
  return risk;
}
