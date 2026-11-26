# Pre-registration (commit BEFORE the final runs)

> DRAFT: edit the predictions so they are your own, then `git commit` this file before running
> `q1a_final`. The commit timestamp is the evidence that the criteria came before the results.

## Fairness criterion for MLP vs CNN (Q1(a), Q2(a))
1. Trainable parameters within ±10%: MLP-B 856,074 vs CNN-A 816,202 (ratio 0.953).
2. Identical data, split, preprocessing and **no augmentation**.
3. Identical optimiser (AdamW, wd 5e-4), batch size 128, epochs, cosine schedule.
4. Identical tuning budget: the LR grid {3e-4, 1e-3, 3e-3}, chosen by validation accuracy, seed 26011.
5. Identical checkpoint rule: best validation accuracy.
6. Three seeds (26011–26013); differences smaller than the seed std are not called differences.
7. Compute is **not** matched: MACs per image are reported (MLP-B 854,528; CNN-A 131,532,672).

## Hypotheses and predictions
- **Q1(a):** H1, the missing spatial inductive bias, is the main reason the MLP does poorly. It predicts
  high MLP training accuracy with a large train–val gap, validation accuracy that saturates as width
  grows, small validation gains from BN/LR tuning, and a higher CNN validation accuracy at matched
  parameters. H2 (under-fitting) predicts low training accuracy and validation accuracy that rises with
  parameters.
- **Q2(b):** a fixed pixel permutation leaves MLP-B unchanged within seed noise and drops CNN-A
  substantially, towards or below MLP-B. If CNN-A keeps a clear advantage, locality is *not* the reason.
- **Q2(c):** E1 (colour statistics drive the CNN) vs E2 (luminance spatial structure). Gray→gray close
  to RGB→RGB supports E2. A large per-class drop concentrated in colour-typical classes supports E1.
- **Q1(b) decision rule:** "structure is learned before memorisation" iff, at t* (the epoch of peak val
  accuracy), the noisy-subset fit to the assigned wrong labels is < 20% and the noisy-subset accuracy
  against the true labels is > 10%.
- **Q3(b):** denoising "recovers the information the classifier needs" if DAE(noisy) accuracy is clearly
  above Noisy accuracy and closes most of the gap to AE(clean), the blur-limited ceiling.

## Fixed in advance
- σ = 0.2 (Q3(b)); 40% label noise (Q1(b)); Q4 representation = AE d = 256; Q4 subset = 10,000.
- Q4 clustering selection rule: highest validation silhouette.
