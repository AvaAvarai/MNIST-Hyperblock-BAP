# BAP: Progressive Batch Gathering

Bidirectional Active Processing (BAP) approach that gathers training cases in batches. Does not run full HB generation each time. Instead uses few steps per run, picks best, seeds next round. Repeats until target accuracy on test.

## Stratified ordering

Build a stratified index order over train data:

- Group indices by class
- Shuffle each class
- Interleave round-robin across classes

Result: first N indices have balanced class distribution.

## Main loop

```
n_fit = 0
seed_indices = empty
seed_hbs = empty

while n_fit < n_max:
    n_fit = n_fit + chunk
    idx = stratified indices for first n_fit points (union with seed_indices if any)
    X_fit, y_fit = train data at idx

    if seed_hbs is empty:
        // First round: widest net, fast exploration
        for t = 1 to n_trials:
            shuffle X_fit, y_fit
            hbs = IMHyper(X, y) in exploration mode
            acc = classify test with hbs, k=3
            add (hbs, acc) to candidates
    else:
        // Later rounds: deep refinement from seeds + exploration
        for each hb_seed in seed_hbs:
            hbs = MHyper(X_fit, y_fit, initial_blocks=hb_seed) in refinement mode
            acc = classify test with hbs, k=3
          q q      add (hbs, acc) to candidates
        for a few extra runs:
            shuffle X_fit, y_fit
            hbs = IMHyper(X, y) in exploration mode
            acc = classify test with hbs, k=3
            add (hbs, acc) to candidates

    sort candidates by acc descending
    seed_hbs = top n_top HB sets from candidates
    seed_indices = idx
    best_hbs, best_acc = top candidate

    if best_acc >= target:
        done

return best_hbs
```

## Modes

- **Exploration**: few steps per run. Fast. Used for first round and fresh runs.
- **Refinement**: more steps per run. Used when running from seeds. Deep enough to capture worsen-then-benefit.

## HB generation and classification

- HB gen: IHyper, MHyper, IMHyper (Huber et al. [2]). See Algos.md and hyperblock_algorithms.py.
- Classification: in-block → block class; outside → k-NN over HBs, k=3 (Snyder et al. [3]).
- HB simplify: redundant attribute removal, redundant block removal via overlap, disjunctive units (Snyder et al. [3]; to be implemented).

See README.md for references [1]–[3].

## Next Steps

find largest HBs
see top 10 size wise
ideally ~4000 cases
visualize 10 pictures
if HBs are pure then 100% accuracy
see what ACC is provided with K-NN with k=3 for other cases

Other path:
use knn for classifier
aim for 1800 - 2300
then build HBs on these cases only

capture majority of cases with HBs then visualize them

binary search take 1/2 then 1/4 then ... splitting best each time.

Find one small dataset, small set of large HBs, then see howm many real cases will be in those HBs.


Vusyakuzagui bthen can viz with GLC-L, all 10 in one plane, then connect HB low edge to top edge with a red line. Do for each HB. Show only this with toggle of 

full lossless viz in GLC-L then can optimize the slope of the red line to make red arrow not overlapping. Order of attributes can also be used to viz distintly these HBs.
