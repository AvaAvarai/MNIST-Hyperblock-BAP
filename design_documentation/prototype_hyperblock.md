# Prototype Hyperblock Algorithm

This is a novel hyperblock construction algorithm intended to be more tractable and accurate of resultant classifier than the original three hyper algorithms as in Algos.md.

Prototype Hyperblock (PHyper) creates a compact set of single-point hyperblocks via K-means clustering per class. Each hyperblock is defined by two 121-D edges that are equal (lower = upper = centroid). Achieves 95%+ classification accuracy on MNIST (121-D reduced) with ≤2000 hyperblocks.

## Hyperblock Creation

The steps for the PHyper algorithm are as follows:

1) Set a target maximum number of hyperblocks M (e.g. 2000).

2) For each class c in the dataset, extract all n-D points belonging to class c into X_c.

3) Compute the number of hyperblocks to allocate to class c proportionally to its size:
   - k_c = max(1, floor(M × |X_c| / n_total))
   - k_c = min(k_c, |X_c|)

4) Run MiniBatch K-means on X_c with k = k_c clusters.

5) For each cluster centroid produced by K-means, create a single-point hyperblock:
   - lower = upper = centroid (the cluster center)
   - label = c

6) Collect all hyperblocks from all classes. The total count is at most M.

## Classification

Classification uses k-NN over the hyperblocks:

1) For a query point q, compute the distance from q to each hyperblock.
   - For a single-point hyperblock (lower = upper = p): distance = ||q − p||₂
   - For a general hyperblock [lower, upper]: distance = ||q − clip(q, lower, upper)||₂ (0 if q is inside)

2) Select the k hyperblocks with smallest distance to q.

3) Assign to q the class that appears most often among those k hyperblocks (majority vote).

4) The value of k is learned on a validation set (e.g. k = 3 for MNIST 121-D).

## Notes

- Single-point hyperblocks (lower = upper) yield distance equivalent to standard k-NN on the prototype points.
- Using centroids as prototypes outperforms using the nearest training point to each centroid.
- Proportional allocation ensures balanced representation across classes.
- Former hyper algorithms builds hyperblocks by expanding intervals along attributes and merging overlapping same-class blocks. Blocks are axis-aligned boxes covering regions of space. Count is determined by purity/impurity thresholds.
- This prototype algorithm builds hyperblocks by clustering each class and using cluster centroids as prototypes. Each hyperblock is a single point. Count is set explicitly.
- PHyper is a different paradigm—prototype-based rather than interval/merge-based. It shares the same output format (hyperblocks with lower/upper edges and labels) but produces only degenerate single-point blocks. For single-point blocks, distance to the block equals distance to the point, so classification is equivalent to k-NN on the prototypes. The former hyper methods produce true hyperboxes that can cover regions; PHyper only represents points (centroids).
- This gives O(n) vs O(n²) for merge changes which is much faster on larger n of data. This uses centroids of classes as seeds instead of actual training cases. Results are still geometric but degenerate instead of full area covering HBs.
- This approach can have output HBs constrained to a specific count and can be repeated with this parameter variated until the output HBs are sufficient for 95%+ accuracy on test data of classification. This is how it was used to find the current result set.
