#!/usr/bin/env python3
"""CPS Exp 2: Deadline-aware DVFS, with thermal dynamics and matched workloads.

Synthetic simulator, not an implementation/reproduction of EarDVFS.
Uses 10-ms discrete steps, a single CPU, periodic soft real-time jobs,
FIFO execution and three frequency states.
"""
import argparse
import random
from dataclasses import dataclass

DT = 0.01
PERIOD = 0.20
DEADLINE = 0.17
FREQS = (1.0, 1.5, 2.0)
THERMAL_LIMIT = 51.5
THERMAL_RECOVER = 48.5

@dataclass
class Job:
    release: float
    deadline: float
    work: float  # GHz * seconds (normalized CPU-work units)
    remaining: float
    counted_miss: bool = False


def workload(seed: int, seconds: int) -> list[Job]:
    """Pre-generated workloads ensure all policies face identical jobs."""
    rng = random.Random(seed)
    jobs = []
    for k in range(round(seconds / PERIOD)):
        release = k * PERIOD
        # Alternating moderate/heavy burst periods; occasional outliers.
        heavy = (k // 24) % 2 == 1
        work = rng.uniform(0.145, 0.190) if not heavy else rng.uniform(0.215, 0.275)
        if rng.random() < 0.06:
            work *= 1.20
        jobs.append(Job(release, release + DEADLINE, work, work))
    return jobs


def choose_freq(policy: str, now: float, temp: float, queue: list[Job], ambient: float) -> float:
    if policy == 'performance':
        return 2.0
    if policy == 'balanced':
        return 1.5
    if policy == 'thermal':
        return 1.0 if temp > 46 else 2.0
    if policy != 'adaptive':
        raise ValueError(policy)

    if not queue:
        return 1.0

    # FIFO queue: total work of all jobs in front of earliest deadline.
    # If queued work cannot meet the earliest deadline even at 2 GHz,
    # choose max speed rather than pretend that a feasible plan exists.
    earliest = min(queue, key=lambda j: j.deadline)
    urgent_work = sum(j.remaining for j in queue if j.deadline <= earliest.deadline + 1e-8)
    slack = max(DT, earliest.deadline - now)
    required = urgent_work / slack

    # Safety margin compensates for discrete decisions and workload perturbations.
    target = required * 1.08
    selected = next((f for f in FREQS if f >= target), FREQS[-1])

    # Thermal headroom: prefer lower feasible speed if nearing limit,
    # but never sacrifice a currently feasible deadline merely to cool down.
    if temp > THERMAL_LIMIT - 2 and selected < 2.0:
        return selected
    return selected


def simulate(policy: str, ambient: float, seed: int, seconds: int) -> dict:
    incoming = workload(seed, seconds)
    jobs = [Job(j.release, j.deadline, j.work, j.remaining) for j in incoming]
    ready: list[Job] = []
    release_idx = 0
    temp = ambient
    throttled = False
    energy = 0.0
    misses = 0
    completed = 0
    max_temp = ambient
    thermal_steps = 0
    trace = []

    def count_overdue(t: float):
        nonlocal misses
        for j in ready:
            if not j.counted_miss and t >= j.deadline - 1e-10 and j.remaining > 1e-10:
                j.counted_miss = True
                misses += 1

    for k in range(round(seconds / DT)):
        now = k * DT
        while release_idx < len(jobs) and jobs[release_idx].release <= now + 1e-10:
            ready.append(jobs[release_idx])
            release_idx += 1

        count_overdue(now)
        requested = choose_freq(policy, now, temp, ready, ambient)

        if temp >= THERMAL_LIMIT:
            throttled = True
        elif temp <= THERMAL_RECOVER:
            throttled = False
        if throttled:
            thermal_steps += 1
        freq = 1.0 if throttled else requested

        # Track busy energy separately from idle energy in the same time step.
        available = DT * freq
        used = 0.0
        while ready and available > 1e-10:
            current = ready[0]
            consumed = min(current.remaining, available)
            current.remaining -= consumed
            available -= consumed
            used += consumed
            if current.remaining <= 1e-10:
                finish = now + used / freq
                if finish > current.deadline + 1e-9 and not current.counted_miss:
                    current.counted_miss = True
                    misses += 1
                ready.pop(0)
                completed += 1

        # Simplified busy/idle power (W).
        busy_fraction = min(1.0, used / (DT * freq))
        busy_power = 0.8 + 2.5 * (freq / 2.0) ** 3
        idle_power = 0.35
        power = busy_fraction * busy_power + (1.0 - busy_fraction) * idle_power
        energy += power * DT
        temp += DT * (2.3 * power - 0.12 * (temp - ambient))
        max_temp = max(max_temp, temp)
        if k % 100 == 0:
            trace.append((round(now, 2), round(temp, 2), freq, len(ready)))

    # Jobs released but not completed when simulation ends count as misses
    # only if their deadline passed. Future deadlines are censored, not failures.
    count_overdue(seconds)
    eligible = sum(j.deadline <= seconds + 1e-9 for j in jobs)
    assert eligible > 0
    return {
        'policy': policy, 'ambient': ambient, 'seed': seed,
        'eligible_jobs': eligible, 'misses': misses,
        'miss_pct': 100 * misses / eligible,
        'completed': completed, 'energy_j': energy,
        'max_temp_c': max_temp,
        'throttle_pct': 100 * thermal_steps / round(seconds / DT),
        'trace': trace,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=int, default=120)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--ambients', type=float, nargs='+', default=[20.0, 30.0, 40.0])
    a = p.parse_args()
    assert a.seconds >= 2 and a.seeds >= 1
    print('ambient policy      miss%   energy(J) maxT(C) throttle%')
    for ambient in a.ambients:
        for policy in ('performance', 'balanced', 'thermal', 'adaptive'):
            outcomes = [simulate(policy, ambient, seed, a.seconds) for seed in range(a.seeds)]
            avg = lambda key: sum(o[key] for o in outcomes) / len(outcomes)
            print(f'{ambient:7.0f} {policy:12s} {avg("miss_pct"):6.2f} {avg("energy_j"):10.2f} '
                  f'{avg("max_temp_c"):7.2f} {avg("throttle_pct"):9.2f}')

if __name__ == '__main__':
    main()
