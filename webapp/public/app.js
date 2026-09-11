/**
 * UI for the AML GNN demo.
 *
 * Loads the exported graph and weights, runs all six variants once on startup,
 * and re-renders from those cached scores. Changing the threshold is pure
 * presentation — no model re-runs — so the slider is instant.
 */

import { runVariant, flatten, scoreSubset } from "./model.js";

const MAX_ROWS = 220;
const NEIGHBOUR_LIMIT = 9;

const state = {
  graph: null,
  variants: [],
  results: new Map(), // variant name -> runVariant output
  activeVariant: null,
  threshold: 0.5,
  filter: "flagged",
  selected: null,
  testIndices: [],
};

/* ------------------------------------------------------------------ load */

async function boot() {
  const [rawGraph, rawModels] = await Promise.all([
    fetch("data/graph.json").then((r) => r.json()),
    fetch("data/models.json").then((r) => r.json()),
  ]);

  state.graph = {
    meta: rawGraph.meta,
    accounts: rawGraph.accounts,
    numNodes: rawGraph.meta.accounts,
    numEdges: rawGraph.meta.transactions,
    nodeFeatures: rawGraph.meta.nodeFeatures,
    edgeFeatures: rawGraph.meta.edgeFeatures,
    x: flatten(rawGraph.x),
    edgeAttr: flatten(rawGraph.edgeAttr),
    edgeIndex: rawGraph.edgeIndex,
    adjRow: rawGraph.adjacency.row,
    adjCol: rawGraph.adjacency.col,
    timeCloseness: Float32Array.from(rawGraph.timeCloseness),
    label: rawGraph.label,
    split: rawGraph.split,
    tx: rawGraph.transactions,
  };

  state.variants = rawModels.variants.map((v) => ({
    ...v,
    weightNode: flatten(v.weightNode),
    weightEdge: flatten(v.weightEdge),
  }));

  const started = performance.now();
  for (const variant of state.variants) {
    state.results.set(variant.name, runVariant(state.graph, variant));
  }
  const elapsed = performance.now() - started;

  // Prefer the variant with the best exported F1 — it makes the best first impression.
  state.activeVariant = state.variants.reduce((best, v) =>
    (v.metrics?.f1 ?? 0) > (best.metrics?.f1 ?? 0) ? v : best,
  );
  state.threshold = state.activeVariant.threshold ?? 0.5;
  state.testIndices = state.graph.split.flatMap((s, i) => (s === 2 ? [i] : []));

  buildNeighbourIndex();
  renderFacts(elapsed);
  renderVariantChips();
  renderFilters();
  wireThreshold();
  refresh({ reselect: true });

  document.getElementById("loading").remove();
  document.getElementById("app").hidden = false;
}

/** Adjacency as per-account lists, so the neighbourhood view is a lookup. */
function buildNeighbourIndex() {
  const { numNodes, numEdges, edgeIndex } = state.graph;
  const index = Array.from({ length: numNodes }, () => []);
  for (let e = 0; e < numEdges; e++) {
    index[edgeIndex[0][e]].push(e);
    index[edgeIndex[1][e]].push(e);
  }
  state.graph.incident = index;
}

/* --------------------------------------------------------------- helpers */

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const fixed = (value, places = 3) => value.toFixed(places);

const money = (value) =>
  value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

const result = () => state.results.get(state.activeVariant.name);
const isFlagged = (edge, variantName = state.activeVariant.name) =>
  state.results.get(variantName).probabilities[edge] >= state.threshold;

/* -------------------------------------------------------------- controls */

function renderFacts(elapsed) {
  const { meta } = state.graph;
  document.getElementById("fact-accounts").textContent = meta.accounts.toLocaleString();
  document.getElementById("fact-transactions").textContent = meta.transactions.toLocaleString();
  document.getElementById("fact-runtime").textContent = elapsed.toFixed(0);
}

function renderVariantChips() {
  const container = document.getElementById("variant-list");
  container.replaceChildren();
  for (const variant of state.variants) {
    const chip = el("button", "chip", variant.name);
    chip.type = "button";
    chip.setAttribute("role", "radio");
    chip.setAttribute("aria-checked", String(variant === state.activeVariant));
    chip.title = describeVariant(variant);
    chip.addEventListener("click", () => {
      state.activeVariant = variant;
      renderVariantChips();
      refresh({ reselect: false });
    });
    container.append(chip);
  }
}

function describeVariant(variant) {
  const decoder =
    variant.decoder === "complex"
      ? "ComplEx: asymmetric, so direction of flow matters"
      : "DistMult: symmetric in sender and receiver";
  if (variant.learnTimeWeight) return `${decoder}. Scaled by recency, with a learned weight.`;
  if (variant.useTime) return `${decoder}. Scaled by how soon it follows the sender's last transaction.`;
  return `${decoder}. No time signal.`;
}

const FILTERS = [
  { id: "flagged", label: "Flagged" },
  { id: "laundering", label: "Actually laundering" },
  { id: "missed", label: "Missed" },
  { id: "alarms", label: "False alarms" },
  { id: "disputed", label: "Variants disagree" },
  { id: "all", label: "All" },
];

function renderFilters() {
  const container = document.getElementById("filters");
  container.replaceChildren();
  for (const filter of FILTERS) {
    const button = el("button", "filter", filter.label);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(filter.id === state.filter));
    button.addEventListener("click", () => {
      state.filter = filter.id;
      renderFilters();
      refresh({ reselect: true });
    });
    container.append(button);
  }
}

function wireThreshold() {
  const slider = document.getElementById("threshold");
  slider.value = String(state.threshold);
  document.getElementById("threshold-value").textContent = fixed(state.threshold, 2);
  slider.addEventListener("input", () => {
    state.threshold = Number(slider.value);
    document.getElementById("threshold-value").textContent = fixed(state.threshold, 2);
    refresh({ reselect: false });
  });
}

/* ------------------------------------------------------------ transactions */

function matchingEdges() {
  const { numEdges, label } = state.graph;
  const { probabilities } = result();
  const edges = [];

  for (let e = 0; e < numEdges; e++) {
    const flagged = probabilities[e] >= state.threshold;
    const fraud = label[e] === 1;
    let keep;
    switch (state.filter) {
      case "flagged":
        keep = flagged;
        break;
      case "laundering":
        keep = fraud;
        break;
      case "missed":
        keep = fraud && !flagged;
        break;
      case "alarms":
        keep = flagged && !fraud;
        break;
      case "disputed":
        keep = state.variants.some((v) => isFlagged(e, v.name)) && state.variants.some((v) => !isFlagged(e, v.name));
        break;
      default:
        keep = true;
    }
    if (keep) edges.push(e);
  }

  edges.sort((a, b) => probabilities[b] - probabilities[a]);
  return edges;
}

function renderTransactions(edges, { reselect }) {
  const body = document.getElementById("tx-body");
  const { accounts, edgeIndex, label, tx } = state.graph;
  const { probabilities } = result();

  body.replaceChildren();
  const shown = edges.slice(0, MAX_ROWS);

  if (reselect || !edges.includes(state.selected)) {
    state.selected = shown[0] ?? null;
  }

  for (const e of shown) {
    const row = el("tr", "tx-row");
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-selected", String(e === state.selected));

    const pair = el("td");
    const wrap = el("span", "pair");
    wrap.append(
      el("span", null, accounts[edgeIndex[0][e]]),
      el("span", "arrow", "→"),
      el("span", null, accounts[edgeIndex[1][e]]),
    );
    pair.append(wrap);

    const amount = el("td", "num", money(tx.amount[e]));
    const method = el("td", "method", tx.format[e]);

    const probability = el("td", "num");
    const value = el("span", `prob${probabilities[e] >= state.threshold ? " flagged" : ""}`, fixed(probabilities[e]));
    const dot = el("span", `truth-dot ${label[e] === 1 ? "fraud" : "clean"}`);
    dot.title = label[e] === 1 ? "Actually laundering" : "Actually clean";
    probability.append(value, dot);

    row.append(pair, amount, method, probability);
    const select = () => {
      state.selected = e;
      refresh({ reselect: false });
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    body.append(row);
  }

  document.getElementById("tx-empty").hidden = edges.length > 0;
  document.getElementById("tx-count").textContent =
    edges.length > MAX_ROWS
      ? `Showing the ${MAX_ROWS} highest-scoring of ${edges.length.toLocaleString()} matches.`
      : `${edges.length.toLocaleString()} transaction${edges.length === 1 ? "" : "s"}. The dot on the right is the ground truth.`;
}

/* --------------------------------------------------------------- inspector */

function renderInspector() {
  const host = document.getElementById("inspect");
  document.getElementById("inspect-variant").textContent = state.activeVariant.name;
  host.replaceChildren();

  if (state.selected === null) {
    host.append(el("p", "placeholder", "Select a transaction to see how the model scores it."));
    return;
  }

  const edge = state.selected;
  const { accounts, edgeIndex, tx, timeCloseness, label, incident } = state.graph;
  const variant = state.activeVariant;
  const { nodeEmbedding, edgeEmbedding, rawScores, logits, probabilities, channels } = result();
  const head = edgeIndex[0][edge];
  const tail = edgeIndex[1][edge];

  host.append(
    step(
      1,
      "The transaction as a triple",
      "The sender and receiver are the head and tail; the transaction's own features are the relation between them.",
      (() => {
        const box = el("div", "triple");
        box.append(
          el("span", "node-pill", accounts[head]),
          el("span", "arrow", "—"),
          el("span", "edge-pill", `${money(tx.amount[edge])} · ${tx.format[edge]}`),
          el("span", "arrow", "→"),
          el("span", "node-pill", accounts[tail]),
        );
        const when = el("p", "step-note", `${tx.timestamp[edge]} · ${tx.currency[edge]}`);
        when.style.marginTop = "8px";
        when.style.marginBottom = "0";
        return wrap(box, when);
      })(),
    ),
  );

  const headDegree = incident[head].length;
  const tailDegree = incident[tail].length;
  host.append(
    step(
      2,
      "Who they bank with",
      `The sender has ${headDegree} transaction${headDegree === 1 ? "" : "s"} on record and the receiver ${tailDegree}. ` +
        "Each account's embedding is built from its counterparties' features, not just its own — that is the whole point of using a graph.",
      wrap(neighbourhoodSvg(edge, head, tail), legend()),
    ),
  );

  host.append(
    step(
      3,
      "Where the score comes from",
      `Each of the ${channels} channels contributes sender × transaction × receiver. Bars above the line push towards laundering, below it towards clean.`,
      wrap(channelChart(nodeEmbedding, edgeEmbedding, head, tail, edge, channels, variant), channelAxis(channels)),
    ),
  );

  const maths = el("div", "maths");
  maths.append(mathsRow("sum over channels", fixed(rawScores[edge], 4)));
  if (variant.useTime) {
    maths.append(mathsRow("× time closeness", fixed(timeCloseness[edge], 4)));
    if (variant.timeWeight !== null && variant.timeWeight !== undefined) {
      maths.append(mathsRow("× learned time weight", fixed(variant.timeWeight, 4)));
    }
    maths.append(mathsRow("= logit", fixed(logits[edge], 4)));
  }
  maths.append(mathsRow("sigmoid(logit)", fixed(probabilities[edge], 4), true));
  host.append(
    step(
      4,
      variant.useTime ? "Scaled by how recent it is" : "Squashed into a probability",
      variant.useTime
        ? "This transaction followed the sender's previous one closely enough to score " +
            `${fixed(timeCloseness[edge], 2)} on recency. Bursts of activity are what layering looks like.`
        : "This variant ignores timing entirely — the raw score goes straight through the sigmoid.",
      maths,
    ),
  );

  host.append(step(5, "Verdict", "", verdict(edge, probabilities[edge], label[edge] === 1)));
}

function wrap(...nodes) {
  const box = el("div");
  box.append(...nodes);
  return box;
}

function step(number, title, note, content) {
  const row = el("div", "step");
  const body = el("div");
  body.append(el("div", "step-title", title));
  if (note) body.append(el("p", "step-note", note));
  body.append(content);
  row.append(el("div", "step-number", String(number)), body);
  return row;
}

function mathsRow(labelText, value, total = false) {
  const row = el("div", `maths-row${total ? " total" : ""}`);
  row.append(el("span", null, labelText), el("span", null, value));
  return row;
}

function channelChart(nodeEmbedding, edgeEmbedding, head, tail, edge, channels, variant) {
  const contributions = channelContributions(nodeEmbedding, edgeEmbedding, head, tail, edge, channels, variant);
  const peak = Math.max(...contributions.map(Math.abs), 1e-6);

  const chart = el("div", "channels");
  for (const [index, value] of contributions.entries()) {
    const column = el("div", `channel ${value >= 0 ? "pos" : "neg"}`);
    const bar = el("div", "bar");
    bar.style.height = `${Math.max(2, (Math.abs(value) / peak) * 42)}px`;
    column.title = `channel ${index}: ${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
    column.append(bar);
    chart.append(column);
  }
  return chart;
}

/**
 * Per-channel contribution to the raw score. For DistMult that is literally
 * h·r·t per channel; for ComplEx the four real/imaginary terms are summed into
 * the channel pair they came from, so the chart still adds up to the score.
 */
function channelContributions(nodeEmbedding, edgeEmbedding, head, tail, edge, channels, variant) {
  const h = head * channels;
  const t = tail * channels;
  const r = edge * channels;

  if (variant.decoder !== "complex") {
    return Array.from(
      { length: channels },
      (_, d) => nodeEmbedding[h + d] * edgeEmbedding[r + d] * nodeEmbedding[t + d],
    );
  }

  const half = channels / 2;
  return Array.from({ length: half }, (_, d) => {
    const hRe = nodeEmbedding[h + d];
    const hIm = nodeEmbedding[h + half + d];
    const rRe = edgeEmbedding[r + d];
    const rIm = edgeEmbedding[r + half + d];
    const tRe = nodeEmbedding[t + d];
    const tIm = nodeEmbedding[t + half + d];
    return hRe * rRe * tRe + hRe * rIm * tIm + hIm * rRe * tIm - hIm * rIm * tRe;
  });
}

function channelAxis(channels) {
  const axis = el("div", "channel-axis");
  axis.append(el("span", null, "channel 0"), el("span", null, `channel ${channels - 1}`));
  return axis;
}

function legend() {
  const box = el("div", "legend");
  const entries = [
    ["this transaction", "var(--accent)"],
    ["laundering", "var(--fraud)"],
    ["clean", "#3a4a5e"],
  ];
  for (const [text, colour] of entries) {
    const item = el("span", null, text);
    item.style.color = colour;
    box.append(item);
  }
  return box;
}

/* ------------------------------------------------------- neighbourhood svg */

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

/**
 * One-hop neighbourhood around the selected transaction, laid out with a few
 * iterations of a spring model. Deliberately capped — a hub account can have
 * hundreds of counterparties and the picture stops meaning anything.
 */
function neighbourhoodSvg(edge, head, tail) {
  const { edgeIndex, label, incident } = state.graph;
  const width = 560;
  const height = 240;

  const nodeIds = new Set([head, tail]);
  for (const anchor of [head, tail]) {
    for (const e of incident[anchor].slice(0, NEIGHBOUR_LIMIT)) {
      nodeIds.add(edgeIndex[0][e]);
      nodeIds.add(edgeIndex[1][e]);
    }
  }

  const ids = [...nodeIds];
  const position = new Map();
  ids.forEach((id, i) => {
    const angle = (2 * Math.PI * i) / ids.length;
    position.set(id, {
      x: width / 2 + Math.cos(angle) * 80 + (id % 7) - 3,
      y: height / 2 + Math.sin(angle) * 62 + (id % 5) - 2,
    });
  });
  position.set(head, { x: width / 2 - 90, y: height / 2 });
  position.set(tail, { x: width / 2 + 90, y: height / 2 });

  const links = [];
  for (const anchor of [head, tail]) {
    for (const e of incident[anchor].slice(0, NEIGHBOUR_LIMIT)) {
      if (nodeIds.has(edgeIndex[0][e]) && nodeIds.has(edgeIndex[1][e])) {
        links.push({ edge: e, a: edgeIndex[0][e], b: edgeIndex[1][e] });
      }
    }
  }

  relax(ids, links, position, width, height, head, tail);

  const svg = svgEl("svg", { class: "neighbourhood", viewBox: `0 0 ${width} ${height}` });
  for (const link of links) {
    const a = position.get(link.a);
    const b = position.get(link.b);
    const selected = link.edge === edge;
    svg.append(
      svgEl("line", {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        stroke: selected ? "var(--accent)" : label[link.edge] === 1 ? "var(--fraud)" : "#2a3647",
        "stroke-width": selected ? 2.4 : 1.1,
        "stroke-opacity": selected ? 1 : 0.75,
      }),
    );
  }
  for (const id of ids) {
    const { x, y } = position.get(id);
    const anchor = id === head || id === tail;
    svg.append(
      svgEl("circle", {
        cx: x,
        cy: y,
        r: anchor ? 7 : 3.6,
        fill: anchor ? "var(--accent)" : "#3a4a5e",
        stroke: "#0c1219",
        "stroke-width": 1.5,
      }),
    );
  }
  for (const [id, dx] of [
    [head, -12],
    [tail, 12],
  ]) {
    const { x, y } = position.get(id);
    svg.append(
      Object.assign(
        svgEl("text", {
          x: x + dx,
          y: y - 13,
          fill: "#b9d6f5",
          "font-size": 10,
          "font-family": "var(--mono)",
          "text-anchor": dx < 0 ? "end" : "start",
        }),
        { textContent: state.graph.accounts[id] },
      ),
    );
  }
  return svg;
}

/** A handful of spring-repulsion passes; enough to stop nodes overlapping. */
function relax(ids, links, position, width, height, head, tail) {
  const pinned = new Set([head, tail]);
  for (let pass = 0; pass < 90; pass++) {
    for (const a of ids) {
      for (const b of ids) {
        if (a === b) continue;
        const pa = position.get(a);
        const pb = position.get(b);
        let dx = pa.x - pb.x;
        let dy = pa.y - pb.y;
        let distance = Math.hypot(dx, dy) || 0.01;
        if (distance < 42) {
          const push = (42 - distance) / distance / 2.4;
          if (!pinned.has(a)) {
            pa.x += dx * push;
            pa.y += dy * push;
          }
        }
      }
    }
    for (const link of links) {
      const pa = position.get(link.a);
      const pb = position.get(link.b);
      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const distance = Math.hypot(dx, dy) || 0.01;
      const pull = ((distance - 70) / distance) * 0.06;
      if (!pinned.has(link.a)) {
        pa.x += dx * pull;
        pa.y += dy * pull;
      }
      if (!pinned.has(link.b)) {
        pb.x -= dx * pull;
        pb.y -= dy * pull;
      }
    }
    for (const id of ids) {
      const p = position.get(id);
      p.x = Math.min(width - 18, Math.max(18, p.x));
      p.y = Math.min(height - 22, Math.max(22, p.y));
    }
  }
}

/* ----------------------------------------------------------------- verdict */

function verdict(edge, probability, isFraud) {
  const flagged = probability >= state.threshold;
  const box = el("div");

  const cards = el("div", "verdict");
  const model = el("div", `verdict-card ${flagged ? "flagged" : "clean"}`);
  model.append(el("label", null, "Model says"), el("strong", null, flagged ? "LAUNDERING" : "CLEAN"));
  const confidence = el("div", "verdict-card");
  confidence.append(el("label", null, "Probability"), el("strong", null, fixed(probability, 4)));
  const truth = el("div", `verdict-card ${isFraud ? "fraud" : "clean"}`);
  truth.append(el("label", null, "Ground truth"), el("strong", null, isFraud ? "LAUNDERING" : "CLEAN"));
  cards.append(model, confidence, truth);

  let outcome;
  if (flagged && isFraud) outcome = ["hit", "Correctly caught. This one is laundering and the model flagged it."];
  else if (!flagged && !isFraud) outcome = ["hit", "Correctly cleared. Nothing to investigate here."];
  else if (flagged && !isFraud)
    outcome = ["alarm", "False alarm. A clean transaction sent for review — the cost of a lower threshold."];
  else outcome = ["miss", "Missed. This is laundering and the model let it through."];

  const banner = el("div", `outcome ${outcome[0]}`, outcome[1]);
  banner.style.marginTop = "10px";
  box.append(cards, banner);
  return box;
}

/* -------------------------------------------------------------- comparison */

function renderComparison() {
  const bars = document.getElementById("variant-bars");
  bars.replaceChildren();

  if (state.selected === null) {
    bars.append(el("p", "placeholder", "Select a transaction."));
    document.getElementById("variant-agreement").textContent = "";
  } else {
    const edge = state.selected;
    let flaggedCount = 0;
    for (const variant of state.variants) {
      const probability = state.results.get(variant.name).probabilities[edge];
      const flagged = probability >= state.threshold;
      if (flagged) flaggedCount++;

      const row = el("div", `variant-bar${variant === state.activeVariant ? " active" : ""}`);
      const track = el("div", "bar-track");
      const fill = el("div", `bar-fill${flagged ? " flagged" : ""}`);
      fill.style.width = `${Math.max(1, probability * 100)}%`;
      const line = el("div", "bar-line");
      line.style.left = `${state.threshold * 100}%`;
      line.title = `threshold ${fixed(state.threshold, 2)}`;
      track.append(fill, line);
      row.append(el("span", null, variant.name), track, el("span", null, fixed(probability)));
      bars.append(row);
    }

    const isFraud = state.graph.label[edge] === 1;
    const note = document.getElementById("variant-agreement");
    if (flaggedCount === state.variants.length) {
      note.textContent = `All six flag this one. It is ${isFraud ? "indeed laundering" : "actually clean, so all six are wrong together"}.`;
    } else if (flaggedCount === 0) {
      note.textContent = `None of the six flag it. It is ${isFraud ? "laundering, so every variant missed it" : "clean — all six agree correctly"}.`;
    } else {
      const hint =
        state.filter === "disputed"
          ? " Nudge the threshold and watch which ones change their mind."
          : " Disagreement like this is where the threshold matters most — try the “Variants disagree” filter.";
      note.textContent =
        `${flaggedCount} of ${state.variants.length} flag this transaction, ${state.variants.length - flaggedCount} clear it. ` +
        `It is actually ${isFraud ? "laundering" : "clean"}.${hint}`;
    }
  }

  const body = document.getElementById("metrics-body");
  body.replaceChildren();
  const rows = state.variants.map((variant) => ({
    variant,
    stats: scoreSubset(
      state.results.get(variant.name).probabilities,
      state.graph.label,
      state.testIndices,
      state.threshold,
    ),
  }));
  const bestF1 = Math.max(...rows.map((r) => r.stats.f1));

  for (const { variant, stats } of rows) {
    const row = el("tr", variant === state.activeVariant ? "active" : "");
    const f1 = el("td", "num", fixed(stats.f1));
    if (stats.f1 === bestF1) f1.classList.add("best");
    row.append(
      el("td", null, variant.name),
      el("td", "num", fixed(stats.precision)),
      el("td", "num", fixed(stats.recall)),
      f1,
      el("td", "num", stats.tp.toLocaleString()),
      el("td", "num", stats.fn.toLocaleString()),
      el("td", "num", stats.fp.toLocaleString()),
    );
    row.style.cursor = "pointer";
    row.addEventListener("click", () => {
      state.activeVariant = variant;
      renderVariantChips();
      refresh({ reselect: false });
    });
    body.append(row);
  }
}

/* ----------------------------------------------------------------- refresh */

function refresh({ reselect }) {
  renderTransactions(matchingEdges(), { reselect });
  renderInspector();
  renderComparison();
}

boot().catch((error) => {
  console.error(error);
  const loading = document.getElementById("loading");
  loading.replaceChildren(
    el("p", null, "Could not load the demo data."),
    el(
      "p",
      null,
      "If you are running this locally, serve the folder over HTTP — fetch() will not read files from disk. Try: python3 -m http.server --directory webapp/public",
    ),
  );
});
