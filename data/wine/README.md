# Wine domain inputs

Source: WineEnthusiast reviews scraped by `zynicide` and published on Kaggle as
`zynicide/wine-reviews` (CC0). Mirrored on HuggingFace as
`james-burton/wine_reviews_kaggle`.

Files in this directory are **generated** — do not edit by hand.

## Regenerate

```bash
pip install datasets numpy
python scripts/build_wine_dataset.py --n 200 --seed 42
```

The script writes `products.jsonl` and `reviews.jsonl` (1:1) into this
directory. Stratified to span ≥8 varieties, ≥6 countries, and roughly
even thirds across price tiers <$25 / $25–$60 / >$60, weighted toward
points ≥88 for richer descriptions.
