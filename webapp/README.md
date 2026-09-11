# Web demo

An interactive page that runs the trained model in the browser. Pick a
transaction, watch the graph neural network score it, and compare what all six
variants make of the same case.

There is no backend. The graph tensors and the learned weight matrices are
downloaded once, about 160 KB gzipped, and the forward pass is plain JavaScript
in [`public/model.js`](public/model.js). Scoring all transactions under all six
variants takes a few milliseconds, so the threshold slider is instant.

## Is it really the model?

Yes, and that is tested rather than asserted:

```bash
python webapp/reference_scores.py   # score everything with PyTorch
node webapp/verify_model.mjs        # check the browser code agrees
```

The two agree to about 1e-5, which is float32 noise.

## Run it locally

`fetch()` will not read files from disk, so serve the folder over HTTP:

```bash
python3 -m http.server 4173 --directory webapp/public
```

Then open <http://localhost:4173>.

## Deploy to Vercel

The site is static, so there is nothing to build.

1. Import the repository at [vercel.com/new](https://vercel.com/new).
2. Set **Root Directory** to `webapp`.
3. Framework preset **Other**, and leave the build command empty.

`vercel.json` points the output at `public/`. Or from the command line:

```bash
cd webapp && vercel --prod
```

## Regenerating the demo bundle

`public/data/` is produced from a trained model and is committed so the site
deploys without a training step. To rebuild it, after retraining or to swap in
a model trained on the real IBM data:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic.csv
python scripts/prepare_dataset.py data/synthetic.csv --out data/demo.csv --fraud-ratio 0.4
python scripts/build_graph.py data/demo.csv --out data/demo-graph.pt

for opts in "" "--use-time" "--use-time --learn-time-weight"; do
  for decoder in distmult complex; do
    python scripts/train.py --graph data/demo-graph.pt --decoder $decoder $opts \
      --epochs 250 --patience 30 --out-channels 20
  done
done

python webapp/export_demo.py
```

To use a model trained on the real dataset instead, point `export_demo.py` at
that graph and those checkpoints:

```bash
python webapp/export_demo.py --graph data/graph.pt --transactions data/balanced.csv --artifacts artifacts
```

Nothing in the page is hard coded to the demo dataset. It reads the variant
list, the channel count and the feature names out of the exported JSON. The one
thing to watch is size, since a graph much beyond 10k transactions starts to
make the initial download unpleasant.

## Files

```
public/index.html   markup
public/styles.css   styling
public/model.js     the forward pass, the same arithmetic as src/amlgnn/models.py
public/app.js       UI: transaction list, walkthrough, variant comparison
public/data/        exported graph and weights (committed)
export_demo.py      writes public/data/ from a graph and checkpoints
reference_scores.py PyTorch scores for the verification test
verify_model.mjs    checks the browser forward pass against PyTorch
```
