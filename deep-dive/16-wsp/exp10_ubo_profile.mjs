
import fs from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { StaticNetFilteringEngine } from "@gorhill/ubo-core";
import inspector from "node:inspector";

const raw = await fs.readFile("easylist.txt", "utf8");
const engine = await StaticNetFilteringEngine.create();

const buildStart = performance.now();
await engine.useLists([{ name: "EasyList", raw }]);

console.log(
  "Load time:",
  (performance.now() - buildStart).toFixed(1),
  "ms"
);

const requests = [];
const origins = [
  "https://news.example/",
  "https://shopping.example/",
  "https://blog.example/",
];

for (let i = 0; i < 10000; i++) {
  const originURL = origins[i % origins.length];

  const url = i % 4 === 0
    ? `https://ads.example/ad/banner${i}.js`
    : i % 4 === 1
    ? `https://cdn.example/assets/app${i}.js`
    : i % 4 === 2
    ? `https://metrics.example/collect?id=${i}`
    : `https://static.example/images/${i}.png`;

  requests.push({
    originURL,
    url,
    type: i % 4 === 3 ? "image" : "script",
  });
}

let blocked = 0;
let excepted = 0;
let noMatch = 0;

// JIT warm-up
for (let i = 0; i < 20000; i++) {
  engine.matchRequest(requests[i % requests.length]);
}

const session = new inspector.Session();
session.connect();

const post = (method, params = {}) =>
  new Promise((resolve, reject) => {
    session.post(method, params, (error, result) => {
      if (error) reject(error);
      else resolve(result);
    });
  });

await post("Profiler.enable");
await post("Profiler.start");

const batches = 20;
const timings = [];

for (let batch = 0; batch < batches; batch++) {
  const start = performance.now();

  for (const request of requests) {
    const result = engine.matchRequest(request);

    if (result === 1) blocked++;
    else if (result === 2) excepted++;
    else noMatch++;
  }

  const elapsed = performance.now() - start;
  timings.push(elapsed);

  console.log(
    `Batch ${batch + 1}: ${elapsed.toFixed(2)} ms`
  );
}

const total = timings.reduce((a, b) => a + b, 0);
const totalRequests = batches * requests.length;

console.log("\n=== SUMMARY ===");
console.log("Requests:", totalRequests);
console.log("Blocked:", blocked);
console.log("Exceptions:", excepted);
console.log("No match:", noMatch);
console.log(
  "Mean time/request:",
  (total * 1000 / totalRequests).toFixed(3),
  "us"
);
console.log(
  "Throughput:",
  Math.round(totalRequests / (total / 1000)),
  "req/s"
);

const { profile } = await post("Profiler.stop");

await fs.writeFile(
  "exp10_match_only.cpuprofile",
  JSON.stringify(profile)
);

session.disconnect();
console.log("Saved match-only CPU profile");