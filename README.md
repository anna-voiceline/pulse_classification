## Training

```
python scripts/train.py \
  --pos-emb data/pos72/positives_emb.npy \
  --neg-emb data/pos72/negatives_emb.npy \
  --pos-texts data/pos72/positives_text.npy \
  --neg-texts data/pos72/negatives_text.npy \
  --hidden-dim 512 \
  --dropout 0.3 \
  --batch-size 32 \
  --epochs 30 \
  --pos-weight-mult 1.5 \
  --beta 2.5 \
  --output-dir results \
  --visualize
```