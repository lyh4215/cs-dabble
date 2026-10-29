#!/usr/bin/env python3
"""NetSP Deep Dive 10 / Exp 1: iROV-inspired local protocol experiment.

Educational, local-only protocol model; NOT the authors' implementation.
No real BGP sessions, routes, networks or RPKI certificates are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import ipaddress
import json
import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    prefix: str
    asn: int


@dataclass(frozen=True)
class ROA:
    prefix: str
    max_length: int
    asn: int


class OriginValidator:
    def __init__(self, roas: list[ROA]):
        self.roas = roas

    def validate(self, route: Route) -> str:
        network = ipaddress.ip_network(route.prefix, strict=True)
        matching = [r for r in self.roas if network.subnet_of(ipaddress.ip_network(r.prefix))]
        if not matching:
            return "not-found"
        return "valid" if any(
            route.asn == r.asn and network.prefixlen <= r.max_length
            for r in matching
        ) else "invalid"


class PrefixOwner:
    """Owner policy + toy MAC signatures for a deterministic lab test.

    HMAC is only a local authenticity stand-in, not iROV PKI/RPKI.
    """
    def __init__(self, secret: bytes, allowed: set[Route]):
        self.secret = secret
        self.allowed = allowed
        self.requests = 0

    def respond(self, route: Route, now: int, lifetime: int = 30) -> dict | None:
        self.requests += 1
        if route not in self.allowed:
            return None
        body = {"prefix": route.prefix, "asn": route.asn, "expires": now + lifetime}
        body["mac"] = sign(self.secret, body)
        return body


def sign(secret: bytes, body: dict) -> str:
    data = f'{body["prefix"]}|{body["asn"]}|{body["expires"]}'.encode()
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def verify(secret: bytes, route: Route, now: int, proof: dict | None) -> bool:
    if not isinstance(proof, dict):
        return False
    try:
        body = {"prefix": str(proof["prefix"]), "asn": int(proof["asn"]),
                "expires": int(proof["expires"])}
        if (body["prefix"], body["asn"]) != (route.prefix, route.asn):
            return False
        if not (now < body["expires"] <= now + 300):
            return False
        return hmac.compare_digest(sign(secret, body), str(proof["mac"]))
    except (KeyError, TypeError, ValueError):
        return False


class InteractiveROV:
    def __init__(self, origin: OriginValidator, owner: PrefixOwner, *,
                 query_probability: float, query_budget: int, seed: int,
                 enable_stapling: bool = True):
        self.origin = origin
        self.owner = owner
        self.query_probability = query_probability
        self.query_budget = query_budget
        self.enable_stapling = enable_stapling
        self.random = random.Random(seed)
        self.used_per_epoch: Counter[int] = Counter()
        self.metrics: Counter[str] = Counter()

    def decide(self, route: Route, now: int, staple: dict | None = None) -> str:
        status = self.origin.validate(route)
        if status != "invalid":
            self.metrics["origin_" + status] += 1
            return "accept"   # not-found follows permissive baseline ROV convention
        if self.enable_stapling and verify(self.owner.secret, route, now, staple):
            self.metrics["staple_accepted"] += 1
            return "accept"
        epoch = now // 10
        if self.used_per_epoch[epoch] >= self.query_budget:
            self.metrics["budget_rejected"] += 1
            return "reject"
        if self.random.random() >= self.query_probability:
            self.metrics["not_sampled"] += 1
            return "reject"
        self.used_per_epoch[epoch] += 1
        self.metrics["owner_queries"] += 1
        response = self.owner.respond(route, now)
        if verify(self.owner.secret, route, now, response):
            self.metrics["owner_authorized"] += 1
            return "accept"
        self.metrics["owner_rejected"] += 1
        return "reject"


def invariant_tests() -> None:
    original = Route("203.0.113.0/24", 64500)
    failover = Route("203.0.113.0/24", 64501)
    attacker = Route("203.0.113.0/24", 64599)
    owner = PrefixOwner(b"fixed-test-secret", {failover})
    validator = OriginValidator([ROA("203.0.113.0/24", 24, 64500)])
    assert validator.validate(original) == "valid"
    assert validator.validate(failover) == "invalid"
    assert validator.validate(attacker) == "invalid"
    assert validator.validate(Route("198.51.100.0/24", 64588)) == "not-found"
    router = InteractiveROV(validator, owner, query_probability=1, query_budget=10, seed=7)
    proof = owner.respond(failover, 10, lifetime=30)
    assert router.decide(failover, 12, staple=proof) == "accept"
    assert not verify(owner.secret, failover, 40, proof), "expiry must be exclusive"
    assert not verify(owner.secret, attacker, 12, proof), "staple must bind ASN"
    tampered = dict(proof, asn=64599)
    assert not verify(owner.secret, attacker, 12, tampered), "tampering invalidates MAC"
    assert router.decide(attacker, 12) == "reject"
    assert router.decide(original, 12) == "accept"
    # Exactly one owner query per 10s epoch at budget=1.
    limited = InteractiveROV(validator, owner, query_probability=1, query_budget=1, seed=9)
    assert limited.decide(attacker, 1) == "reject"
    assert limited.decide(failover, 2) == "reject"
    assert limited.metrics["owner_queries"] == 1
    assert limited.metrics["budget_rejected"] == 1
    # Stapling bypasses query-budget denial only with a verified owner proof.
    assert limited.decide(failover, 2, staple=proof) == "accept"
    print("[tests] prefix validation, token binding, tampering, expiry, rate limit: PASS")


def experiment(attack_count: int, p: float, budget: int, seed: int, stapling: bool) -> dict:
    original = Route("203.0.113.0/24", 64500)
    failover = Route("203.0.113.0/24", 64501)
    attacker = Route("203.0.113.0/24", 64599)
    owner = PrefixOwner(b"local-experiment-secret", {failover})
    validator = OriginValidator([ROA("203.0.113.0/24", 24, 64500)])
    router = InteractiveROV(validator, owner, query_probability=p,
                            query_budget=budget, seed=seed, enable_stapling=stapling)
    proof = owner.respond(failover, 0, lifetime=30) if stapling else None
    accepted_legitimate = 0
    accepted_attacker = 0
    # Constant prefix/owner. Attack and legitimate updates are shuffled within each epoch.
    rng = random.Random(seed + 100)
    events = [(failover, 1)] * 20 + [(attacker, 0)] * attack_count
    rng.shuffle(events)
    for route, legitimate in events:
        result = router.decide(route, now=1, staple=proof if legitimate else None)
        if result == "accept":
            accepted_legitimate += legitimate
            accepted_attacker += 1 - legitimate
    assert router.decide(original, 2) == "accept"
    return {
        "p": p, "budget": budget, "stapling": stapling,
        "attack_updates": attack_count,
        "legitimate_accept_percent": round(100 * accepted_legitimate / 20, 1),
        "attacker_accept_count": accepted_attacker,
        "owner_queries": router.metrics["owner_queries"],
        "budget_rejected": router.metrics["budget_rejected"],
        "staple_accepted": router.metrics["staple_accepted"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    invariant_tests()
    scenarios = [
        #p, budget, stapling, label
        (1.0, 1000, False, "Interactive, no limits"),
        (1.0, 10, False, "Budget only"),
        (0.2, 10, False, "Sampling + budget"),
        (0.2, 10, True, "Sampling + budget + stapling"),
    ]
    results = []
    print(f'\nLocal synthetic experiment: {args.attacks} invalid attacker updates, 20 legitimate emergency updates')
    print(f'{"Policy":32} {"Legit%":>8} {"Attack accepted":>15} {"Owner queries":>13} {"Staples":>8}')
    for p, budget, stapling, label in scenarios:
        row = experiment(args.attacks, p, budget, args.seed, stapling)
        row["policy"] = label
        results.append(row)
        print(f'{label:32} {row["legitimate_accept_percent"]:8.1f} {row["attacker_accept_count"]:15} '
              f'{row["owner_queries"]:13} {row["staple_accepted"]:8}')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
    print('\nNote: synthetic local protocol model; no real BGP traffic and no paper result reproduction.')


if __name__ == "__main__":
    main()
