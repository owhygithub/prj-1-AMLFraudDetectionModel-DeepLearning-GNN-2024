# Results

Six variants, one graph, one held out test set. Every number here comes from
the original June 2024 runs and is copied straight out of
[results/runs.csv](results/runs.csv). Nothing was rerun after the 2026 code
cleanup. [Known bugs](#known-bugs-in-the-original-runs) covers what that is
worth.

The old files spell the first decoder `DisMult`. The current code spells it
`DistMult`. Same model.

## Setup

| | |
| --- | --- |
| Data | IBM Synthetic AML, `HI-Large_Trans.csv` |
| After balancing | 500,000 transactions, 40% laundering |
| Split | 60% train, 20% validation, 20% test, stratified |
| Test set | 100,000 transactions, 40,000 of them laundering |
| Selection | 5 fold cross validation, early stopping on validation loss |
| Repeats | 7 to 8 runs per variant |
| Threshold | 0.6 |

Numbers below are the mean and standard deviation across repeats. The spread
lands in the fourth decimal because every repeat shared one fixed split and
differed only in weight initialisation.

## Main results

| Variant | Runs | Accuracy | Precision | Recall | F1 | ROC-AUC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DisMult | 7 | 0.7547 ± 0.0004 | 0.8474 ± 0.0012 | 0.4717 ± 0.0015 | 0.6060 ± 0.0010 | 0.7075 ± 0.0005 | 0.4685 ± 0.0066 |
| DisMult-T | 7 | 0.8803 ± 0.0005 | 0.8343 ± 0.0009 | **0.8745** ± 0.0018 | 0.8539 ± 0.0007 | 0.8793 ± 0.0007 | 0.6060 ± 0.0003 |
| DisMult-T+W | 7 | 0.8771 ± 0.0010 | 0.8341 ± 0.0012 | 0.8649 ± 0.0016 | 0.8492 ± 0.0013 | 0.8751 ± 0.0011 | 0.6163 ± 0.0004 |
| ComplEx | 8 | 0.8805 ± 0.0013 | 0.8330 ± 0.0019 | **0.8772** ± 0.0013 | **0.8545** ± 0.0015 | **0.8800** ± 0.0013 | 0.6034 ± 0.0007 |
| ComplEx-T | 8 | **0.8810** ± 0.0009 | 0.8445 ± 0.0011 | 0.8611 ± 0.0016 | 0.8527 ± 0.0011 | 0.8777 ± 0.0010 | 0.6017 ± 0.0005 |
| ComplEx-T+W | 8 | 0.8802 ± 0.0009 | **0.8500** ± 0.0012 | 0.8506 ± 0.0021 | 0.8503 ± 0.0012 | 0.8753 ± 0.0010 | 0.6034 ± 0.0003 |

Hyper-parameters were fixed per decoder after an Optuna search. DistMult used
25 channels and weight decay 5e-4. ComplEx used 20 channels and 5e-5. Both used
lr 0.01, dropout 0.1 and up to 100 epochs.

```bash
python scripts/aggregate_results.py --since 20240625153000 --markdown
```

### Confusion matrices

Worked back from the mean precision and recall, over 100,000 test transactions
of which 40,000 are laundering.

| Variant | Caught | False alarms | Missed | Correctly cleared | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| DisMult | 18,868 | 3,398 | 21,132 | 56,602 | 22,266 |
| DisMult-T | 34,980 | 6,948 | 5,020 | 53,052 | 41,928 |
| DisMult-T+W | 34,595 | 6,881 | 5,405 | 53,119 | 41,476 |
| ComplEx | 35,088 | 7,034 | 4,912 | 52,966 | 42,122 |
| ComplEx-T | 34,445 | 6,342 | 5,555 | 53,658 | 40,787 |
| ComplEx-T+W | 34,024 | 6,005 | 5,976 | 53,995 | 40,030 |

The five working variants all flag about 41,000 transactions to catch about
34,500 of the 40,000. Plain `DisMult` flags half as many and misses more than
half the laundering.

Figures for each variant are in [results/figures/](results/figures/).

| ROC curve | Confusion matrix |
| --- | --- |
| ![ComplEx ROC](results/figures/ComplEx/test-roc-curve.png) | ![ComplEx confusion matrix](results/figures/ComplEx/test-confusion-matrix.png) |

## What the numbers say

**The five working variants are the same.** F1 runs from 0.8492 to 0.8545 and
ROC-AUC from 0.8751 to 0.8800. Gaps of about 0.005, five times the run to run
spread, so not noise but not enough to change a decision either. Neither the
decoder nor the time signal moved anything. The
[known bugs](#known-bugs-in-the-original-runs) explain why. With a zero
imaginary part ComplEx *is* DistMult, and with a detached weight `-T+W` *is*
`-T`. Three of the six were never separate models.

**There is a precision and recall trade to pick from.** `ComplEx` catches the
most laundering at recall 0.877. `ComplEx-T+W` raises the fewest false alarms
at precision 0.850. In AML triage a missed case costs more than an extra
review, so the recall end is the one to take.

**Plain `DisMult` is not a fair comparison.** Its script was the only one with
the output sigmoid commented out, so the threshold of 0.6 hit a raw score
instead of a probability. That alone flattens its recall. Its ROC-AUC is also
the worst at 0.707 against roughly 0.88, and ROC-AUC ignores the threshold, so
something else is going on too. Unbounded raw scores fed into the loss look
like they trained less stably than the accidentally squashed ones. On the
smaller pilot the same setup reached 0.837, which fits that reading without
proving it.

**MRR as logged means nothing here.** It sits near 0.0002 for every variant.
The implementation averages `1/rank` over all 40,000 positives in a 100,000
edge ranking, so its scale is set by how many positives there are rather than
by the model. The current code keeps it and adds average precision, which is
the one to read.

## Pilot round

An earlier round on a much smaller graph, 2,000 test transactions. Single runs,
so no error bars.

| Variant | Channels | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DisMult | 20 | 200 | 0.8795 | 0.8763 | 0.6967 | 0.7762 | 0.8273 |
| DisMult | 30 | 200 | 0.8840 | 0.8710 | 0.7200 | 0.7883 | 0.8371 |
| DisMult | 30 | 100 | 0.8860 | 0.8470 | 0.7567 | 0.7993 | 0.8490 |
| DisMult-T | 20 | 200 | 0.8900 | 0.8345 | 0.7900 | 0.8116 | 0.8614 |
| ComplEx | 15 | 100 | 0.8885 | 0.7878 | 0.8600 | 0.8223 | 0.8804 |
| ComplEx-T | 15 | 100 | 0.8765 | 0.7754 | 0.8283 | 0.8010 | 0.8627 |
| ComplEx-T+W | 15 | 100 | 0.8835 | 0.7845 | 0.8433 | 0.8129 | 0.8720 |

Accuracy barely moved going to the 50 times larger run, but recall improved a
lot. The graph got denser rather than just bigger, so each account carries more
neighbourhood context.

## Known bugs in the original runs

Found while cleaning up the code in 2026. All are fixed in `src/amlgnn/`. None
are fixed in the numbers above.

**1. Double sigmoid in the loss.**

```python
normalized_scores = torch.sigmoid(raw_scores)   # in the decoder
criterion = nn.BCEWithLogitsLoss()              # sigmoids again
```

`BCEWithLogitsLoss` wants logits. Handing it probabilities squeezes every
prediction into `(0.5, 0.731)` and puts a floor of about 0.54 on the loss. The
recorded losses sit at 0.60 to 0.62, just above it. It also shrinks the
gradient by up to 16x and discards the stable log-sum-exp the fused loss exists
to provide. Hit five of the six variants.

**2. ComplEx was DistMult.**

```python
heads = torch.complex(heads, torch.zeros_like(heads))   # imaginary part = 0
```

With a zero imaginary part `conj` does nothing and the whole expression
collapses to DistMult. The asymmetry ComplEx is chosen for was never there.

**3. The learnable time weight never learned.**

```python
learnable_weight_tensor = torch.tensor(self.learnable_weight, ...)
```

Wrapping a `nn.Parameter` in `torch.tensor` copies it out of the autograd
graph. It got no gradient and held its random starting value for every epoch.
So `-T+W` was `-T` with a random scale bolted on, which fits `-T+W` landing
just below `-T` every single time.

**4. Test edges leaked into cross validation.**

```python
for fold, (train_idx, val_idx) in enumerate(kf.split(range(input_data.edge_attr.shape[0]))):
```

`KFold` ran over every edge in the graph rather than the training split, so
each fold trained on about a fifth of the test edges before being scored on
them. The reported test numbers are optimistic by an unknown amount.

**5. Node features were not reproducible.**

Account identifiers went through Python's `hash()`, which is salted per
process, and every node also got a `random.random()` column as a unique ID. So
node features changed on every run and one of them was pure noise.

### Still not fixed

- Min-max scaling is fitted over the whole dataset at build time, so test
  statistics reach the training features. Mild, but it is leakage.
- The adjacency covers every edge including test edges, so the model knows a
  test transaction exists while it learns. Normal for transductive link
  prediction, worth stating.
- One convolution layer, so each account only sees its immediate neighbours.
  Layering and smurfing span several hops.

## Rerunning

The numbers above cannot be reproduced from this repo as it stands. The fixes
change the model, and the balanced 500k CSV was built on a cluster and not
kept.

```bash
python scripts/prepare_dataset.py data/HI-Large_Trans.csv --out data/balanced.csv --fraud-ratio 0.4
python scripts/build_graph.py data/balanced.csv --out data/graph.pt
python scripts/train.py --graph data/graph.pt --decoder complex --use-time --learn-time-weight --tune
```

Expect different numbers. The fixes pull both ways. Closing the CV leak should
lower them, while a real ComplEx decoder and a working gradient should raise
them.

## Worth trying next

1. Bound the score on purpose. The accidental sigmoid seems to have helped, so
   batch norm on the score or a tanh would test that properly.
2. A second convolution layer, so the model can see the chains laundering
   actually makes.
3. Train on the real class balance. `pos_weight` handles the raw 0.1% rate
   without throwing away 99% of the clean transactions.
4. Report precision at a fixed alert budget. How many real cases land in the
   top 1,000 flags is what an AML team actually works from.
