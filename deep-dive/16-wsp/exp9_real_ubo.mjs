
import { performance } from "node:perf_hooks";
import { StaticNetFilteringEngine } from "@gorhill/ubo-core";

const snfe = await StaticNetFilteringEngine.create();

// 실제 Adblock 필터 문법을 사용하는 소규모 테스트 목록.
// 예시 도메인들은 테스트용이며 네트워크 요청은 보내지 않는다.
const filters = [
  "||ads.example^",
  "||tracker.example^$third-party",
  "||media.example^$image",
  "@@||ads.example/allowed.js",
  "||scripts.example^$script",
];

const cases = [
  {
    name: "Ad domain",
    originURL: "https://news.example/",
    url: "https://ads.example/banner.js",
    type: "script",
  },
  {
    name: "Allowed exception",
    originURL: "https://news.example/",
    url: "https://ads.example/allowed.js",
    type: "script",
  },
  {
    name: "Third-party tracker",
    originURL: "https://news.example/",
    url: "https://tracker.example/pixel",
    type: "image",
  },
  {
    name: "First-party tracker",
    originURL: "https://tracker.example/",
    url: "https://tracker.example/pixel",
    type: "image",
  },
  {
    name: "Image resource",
    originURL: "https://news.example/",
    url: "https://media.example/photo.png",
    type: "image",
  },
  {
    name: "Same URL as script",
    originURL: "https://news.example/",
    url: "https://media.example/photo.png",
    type: "script",
  },
  {
    name: "Normal resource",
    originURL: "https://news.example/",
    url: "https://cdn.example/app.js",
    type: "script",
  },
];

console.log("=== FILTER COMPILATION ===");

const t0 = performance.now();

await snfe.useLists([
  {
    name: "wsp-exp9",
    raw: filters.join("\n"),
  },
]);

console.log(
  `Rules: ${filters.length}`
);
console.log(
  `Compile and load: ${(performance.now() - t0).toFixed(3)} ms`
);

console.log("\n=== DECISION TEST ===");

for (const test of cases) {
  const result = snfe.matchRequest(test);

  console.log(`\n${test.name}`);
  console.log(`  Result: ${result}`);
  console.log(
    `  Decision: ${result === 0 ? "NO MATCH" : "MATCH"}`
  );

  if (result !== 0) {
    console.log("  Matched filter:", snfe.toLogData());
  }
}

// Benchmark uses a fixed, bounded workload.
// No requests are sent over the network.
const iterations = 10000;
const samples = [];

console.log("\n=== LATENCY BENCHMARK ===");

// Warm up the engine and JIT compiler.
for (let i = 0; i < 2000; i++) {
  snfe.matchRequest(cases[i % cases.length]);
}

for (let i = 0; i < iterations; i++) {
  const request = cases[i % cases.length];

  const start = performance.now();
  snfe.matchRequest(request);
  const duration = performance.now() - start;

  samples.push(duration * 1000);
}

samples.sort((a, b) => a - b);

function percentile(p) {
  const index = Math.floor(
    (samples.length - 1) * p
  );
  return samples[index];
}

const mean = samples.reduce((a, b) => a + b, 0)
  / samples.length;

console.log(`Requests: ${iterations}`);
console.log(`Mean: ${mean.toFixed(3)} us`);
console.log(`P95: ${percentile(0.95).toFixed(3)} us`);
console.log(`P99: ${percentile(0.99).toFixed(3)} us`);

console.log("\n=== MEMORY ===");
console.log(
  `Heap used: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MiB`
);
