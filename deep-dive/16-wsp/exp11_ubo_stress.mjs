
import fs from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { StaticNetFilteringEngine } from "@gorhill/ubo-core";

const raw = await fs.readFile("easylist.txt", "utf8");
const engine = await StaticNetFilteringEngine.create();

const LOAD_START = performance.now();
await engine.useLists([{ name: "EasyList", raw }]);
const loadMs = performance.now() - LOAD_START;

// Deterministic pseudo-random generator
let state = 42;
function rand() {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state / 4294967296;
}

function pick(items) {
  return items[Math.floor(rand() * items.length)];
}

// Extract candidate words from network-filter rules.
// These are lexical fragments, not necessarily uBO index keys.
function extractTokens(text) {
  const tokens = new Set();

  for (const line of text.split("\n")) {
    const s = line.trim();

    if (
      !s ||
      s.startsWith("!") ||
      s.startsWith("[") ||
      s.startsWith("##") ||
      s.startsWith("#@#")
    ) continue;

    const pattern = s.split("$")[0];

    for (const token of pattern.toLowerCase()
      .matchAll(/[a-z][a-z0-9]{5,13}/g)) {
      tokens.add(token[0]);
    }
  }

  return [...tokens];
}

const filterTokens = extractTokens(raw);
console.log("=== FILTER LIST ===");
console.log("Load time:", loadMs.toFixed(2), "ms");
console.log("Extracted tokens:", filterTokens.length);

if (filterTokens.length === 0) {
  throw new Error("No filter tokens extracted");
}

const baseURL =
  "https://cdn.example.test/resource?data=";

function randomText(length) {
  const chars = "0123456789abcdef";
  let out = "";

  for (let i = 0; i < length; i++) {
    out += chars[Math.floor(rand() * chars.length)];
  }
  return out;
}

function makeURL(kind, length) {
  const target = length - baseURL.length;

  if (target <= 0) throw new Error("URL too short");

  let data = "";

  if (kind === "random") {
    data = randomText(target);
  } else if (kind === "filter-rich") {
    while (data.length < target) {
      data += pick(filterTokens) + "_";
    }
    data = data.slice(0, target);
  } else if (kind === "repeated") {
    data = "bannerads_".repeat(
      Math.ceil(target / 10)
    ).slice(0, target);
  } else {
    throw new Error(`Unknown kind: ${kind}`);
  }

  const url = baseURL + data;

  if (url.length !== length) {
    throw new Error("Length mismatch");
  }

  return url;
}

const KINDS = ["random", "filter-rich", "repeated"];
const LENGTHS = [256, 1024, 4096];
const SAMPLE_COUNT = 120;
const ROUNDS = 12;

const context = {
  originURL: "https://news.example.test/",
  type: "script",
};

const workloads = [];

for (const length of LENGTHS) {
  for (const kind of KINDS) {
    const requests = Array.from(
      { length: SAMPLE_COUNT },
      () => ({
        ...context,
        url: makeURL(kind, length),
      })
    );

    workloads.push({
      kind,
      length,
      requests,
      times: [],
      counts: { block: 0, allow: 0, none: 0 },
    });
  }
}

function classify(result, counts) {
  if (result === 1) counts.block++;
  else if (result === 2) counts.allow++;
  else counts.none++;
}

function percentile(sorted, fraction) {
  const index = Math.floor(
    (sorted.length - 1) * fraction
  );
  return sorted[index];
}

// Warm-up all workload categories
for (let repeat = 0; repeat < 3; repeat++) {
  for (const workload of workloads) {
    for (const request of workload.requests) {
      engine.matchRequest(request);
    }
  }
}

// Rotate order to reduce systematic ordering bias.
for (let round = 0; round < ROUNDS; round++) {
  for (let offset = 0; offset < workloads.length; offset++) {
    const workload = workloads[
      (offset + round) % workloads.length
    ];

    for (const request of workload.requests) {
      const start = performance.now();
      const result = engine.matchRequest(request);
      const elapsed = performance.now() - start;

      workload.times.push(elapsed * 1000);

      // Count decisions only in first measured round.
      if (round === 0) {
        classify(result, workload.counts);
      }
    }
  }
}

console.log("\n=== RESULTS ===");
console.log(
  "Kind".padEnd(14),
  "Length".padStart(7),
  "Mean us".padStart(11),
  "P95 us".padStart(11),
  "P99 us".padStart(11),
  "Blocked".padStart(9),
);

const summary = [];

for (const w of workloads) {
  const times = [...w.times].sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0)
    / times.length;

  const row = {
    kind: w.kind,
    length: w.length,
    mean_us: mean,
    p95_us: percentile(times, 0.95),
    p99_us: percentile(times, 0.99),
    ...w.counts,
  };

  summary.push(row);

  console.log(
    row.kind.padEnd(14),
    String(row.length).padStart(7),
    row.mean_us.toFixed(3).padStart(11),
    row.p95_us.toFixed(3).padStart(11),
    row.p99_us.toFixed(3).padStart(11),
    String(row.block).padStart(9),
  );
}

console.log("\n=== FILTER-RICH / RANDOM ===");

for (const length of LENGTHS) {
  const a = summary.find(
    x => x.length === length && x.kind === "random"
  );
  const b = summary.find(
    x => x.length === length && x.kind === "filter-rich"
  );

  console.log(
    `${length} chars: ${(b.mean_us / a.mean_us).toFixed(2)}x`
  );
}

await fs.writeFile(
  "exp11_results.json",
  JSON.stringify({
    easylist_bytes: raw.length,
    load_ms: loadMs,
    tokens: filterTokens.length,
    sample_count: SAMPLE_COUNT,
    rounds: ROUNDS,
    summary,
  }, null, 2)
);

console.log("\nSaved exp11_results.json");
