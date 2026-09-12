/**
 * Check that the browser forward pass agrees with PyTorch.
 *
 * The demo's claim is that it runs the real model, not an approximation, so
 * that claim is worth testing. Regenerate the reference with:
 *
 *     python webapp/reference_scores.py
 *
 * then run:
 *
 *     node webapp/verify_model.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runVariant, flatten } from "./public/model.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (relative) => JSON.parse(fs.readFileSync(path.join(here, relative), "utf8"));

const raw = read("public/data/graph.json");
const models = read("public/data/models.json");
const referencePath = path.join(here, "public/data/reference-scores.json");
if (!fs.existsSync(referencePath)) {
  console.error("missing reference-scores.json. Run `python webapp/reference_scores.py` first.");
  process.exit(2);
}
const reference = JSON.parse(fs.readFileSync(referencePath, "utf8"));

const graph = {
  numNodes: raw.meta.accounts,
  numEdges: raw.meta.transactions,
  nodeFeatures: raw.meta.nodeFeatures,
  edgeFeatures: raw.meta.edgeFeatures,
  x: flatten(raw.x),
  edgeAttr: flatten(raw.edgeAttr),
  adjRow: raw.adjacency.row,
  adjCol: raw.adjacency.col,
  edgeIndex: raw.edgeIndex,
  timeCloseness: Float32Array.from(raw.timeCloseness),
};

// float32 arithmetic in two runtimes will not agree bit for bit. 1e-4 is far
// tighter than anything that could change a prediction.
const TOLERANCE = 1e-4;
let worst = 0;

for (const variant of models.variants) {
  const prepared = {
    ...variant,
    weightNode: flatten(variant.weightNode),
    weightEdge: flatten(variant.weightEdge),
  };
  const started = performance.now();
  const { probabilities } = runVariant(graph, prepared);
  const elapsed = performance.now() - started;

  const truth = reference[variant.name];
  if (!truth) {
    console.error(`no reference scores for ${variant.name}`);
    process.exit(2);
  }
  let maxDiff = 0;
  for (let i = 0; i < truth.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(probabilities[i] - truth[i]));
  }
  worst = Math.max(worst, maxDiff);
  const status = maxDiff < TOLERANCE ? "ok  " : "FAIL";
  console.log(
    `${status} ${variant.name.padEnd(14)} max|js - pytorch| = ${maxDiff.toExponential(2)}   ${elapsed.toFixed(1)} ms`,
  );
}

console.log(`\nworst disagreement: ${worst.toExponential(2)} (tolerance ${TOLERANCE.toExponential(0)})`);
process.exit(worst < TOLERANCE ? 0 : 1);
