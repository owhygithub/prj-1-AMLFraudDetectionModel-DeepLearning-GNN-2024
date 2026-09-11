/**
 * The AML GNN forward pass, in the browser.
 *
 * The arithmetic is the same as `src/amlgnn/models.py`, run on the same
 * learned weights, rather than an approximation of it:
 *
 *     Zx = (A @ X) @ W_node          account embeddings
 *     Ze = E @ W_edge                transaction embeddings
 *     s  = decode(Zx[head], Ze, Zx[tail])
 *     s  = s * timeCloseness * timeWeight     (the -T / -T+W variants)
 *     p  = sigmoid(s)
 *
 * Dropout is training-only and is skipped here, exactly as `model.eval()` does.
 *
 * Cost for the shipped graph (1,200 accounts, 5,922 transactions, 20 channels):
 * roughly 700k multiply-adds, or a couple of milliseconds. Scoring every
 * transaction under all six variants is cheap enough to redo on demand.
 */

/** Sum each node's neighbours' feature vectors: `out = A @ X` for sparse binary A. */
export function aggregateNeighbours(x, numNodes, numFeatures, adjRow, adjCol) {
  const out = new Float32Array(numNodes * numFeatures);
  for (let k = 0; k < adjRow.length; k++) {
    const target = adjRow[k] * numFeatures;
    const source = adjCol[k] * numFeatures;
    for (let j = 0; j < numFeatures; j++) out[target + j] += x[source + j];
  }
  return out;
}

/** Dense `[rows x inner] @ [inner x cols]`, both operands row-major and flat. */
export function matmul(a, rows, inner, b, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const aOffset = i * inner;
    const outOffset = i * cols;
    for (let k = 0; k < inner; k++) {
      const aValue = a[aOffset + k];
      if (aValue === 0) continue;
      const bOffset = k * cols;
      for (let j = 0; j < cols; j++) out[outOffset + j] += aValue * b[bOffset + j];
    }
  }
  return out;
}

/** Flatten a JSON array-of-arrays into the row-major layout matmul expects. */
export function flatten(rowsOfValues) {
  const rows = rowsOfValues.length;
  const cols = rowsOfValues[0].length;
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) out.set(rowsOfValues[i], i * cols);
  return out;
}

export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

/**
 * `<h, r, t>`, the sum over channels of the three vectors multiplied together.
 * Symmetric in head and tail, which is what ComplEx exists to fix.
 */
export function distmultScore(head, relation, tail, offsetH, offsetR, offsetT, channels) {
  let total = 0;
  for (let d = 0; d < channels; d++) {
    total += head[offsetH + d] * relation[offsetR + d] * tail[offsetT + d];
  }
  return total;
}

/**
 * `Re<h, r, conj(t)>` with the first half of each embedding read as the real
 * part and the second half as the imaginary part. Asymmetric: swapping head
 * and tail flips the sign of the last term, so "A paid B" scores differently
 * from "B paid A".
 */
export function complexScore(head, relation, tail, offsetH, offsetR, offsetT, channels) {
  const half = channels / 2;
  let total = 0;
  for (let d = 0; d < half; d++) {
    const hRe = head[offsetH + d];
    const hIm = head[offsetH + half + d];
    const rRe = relation[offsetR + d];
    const rIm = relation[offsetR + half + d];
    const tRe = tail[offsetT + d];
    const tIm = tail[offsetT + half + d];
    total += hRe * rRe * tRe + hRe * rIm * tIm + hIm * rRe * tIm - hIm * rIm * tRe;
  }
  return total;
}

/**
 * Run one variant over the whole graph.
 *
 * Returns the embeddings as well as the scores, because the demo shows the
 * intermediate values rather than just the verdict.
 */
export function runVariant(graph, variant) {
  const { numNodes, numEdges, nodeFeatures, edgeFeatures, x, edgeAttr, adjRow, adjCol, edgeIndex, timeCloseness } =
    graph;
  const channels = variant.outChannels;

  const aggregated = aggregateNeighbours(x, numNodes, nodeFeatures, adjRow, adjCol);
  const nodeEmbedding = matmul(aggregated, numNodes, nodeFeatures, variant.weightNode, channels);
  const edgeEmbedding = matmul(edgeAttr, numEdges, edgeFeatures, variant.weightEdge, channels);

  const decode = variant.decoder === "complex" ? complexScore : distmultScore;
  const gate = variant.timeWeight ?? 1;

  const rawScores = new Float32Array(numEdges);
  const logits = new Float32Array(numEdges);
  const probabilities = new Float32Array(numEdges);

  for (let e = 0; e < numEdges; e++) {
    const head = edgeIndex[0][e] * channels;
    const tail = edgeIndex[1][e] * channels;
    const raw = decode(nodeEmbedding, edgeEmbedding, nodeEmbedding, head, e * channels, tail, channels);
    rawScores[e] = raw;
    logits[e] = variant.useTime ? raw * timeCloseness[e] * gate : raw;
    probabilities[e] = sigmoid(logits[e]);
  }

  return { nodeEmbedding, edgeEmbedding, rawScores, logits, probabilities, channels };
}

/** Precision / recall / F1 and the confusion matrix over a subset of edges. */
export function scoreSubset(probabilities, labels, indices, threshold) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (const e of indices) {
    const predicted = probabilities[e] >= threshold;
    const actual = labels[e] === 1;
    if (predicted && actual) tp++;
    else if (predicted && !actual) fp++;
    else if (!predicted && actual) fn++;
    else tn++;
  }
  const total = tp + fp + fn + tn || 1;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    tp,
    fp,
    fn,
    tn,
    accuracy: (tp + tn) / total,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
  };
}
