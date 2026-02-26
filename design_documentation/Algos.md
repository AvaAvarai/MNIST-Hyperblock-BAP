# Hyperblock algorithms

Hyperblocks (HBs) are useful for geometrically interpretable classifiers. These Hyper algorithms are available in the publication at: [https://arxiv.org/pdf/2305.18429](https://arxiv.org/pdf/2305.18429).

## Interval Hyper

The first algorithm is Interval Hyper (IHyper). The idea of IHyper is that a hyperblock can be created by repeatedly finding the largest interval of values for some attribute x_i with a purity above a given threshold.

The steps for IHyper algorithm are as follows:

1) For each attribute x_i in a dataset, create an ascending sorted array containing all values for the attribute.
2) Seed value a_1, the first value in the first sorted array and compute LDF G(a) for the n-D point a, which corresponds to a_1. The first sorted array is an array of values of the first attribute. Note, instead of a_1 any value of any sorted array of x_i can be taken.
3) Initialize b_i = a_i = d_i for a_i
4) Create HB for a such that b_1 ≤ a_1 ≤ d_1 & b_2 ≤ a_2 ≤ d_2 & … & b_n ≤ a_n ≤ d_n.
5) Use the next value e_i in the same sorted array to expand the interval on the same attribute if the n-D point e that corresponds with e_i is either of the same class as a or that the interval on this attribute will remain above some purity threshold T despite adding e_i.
6) Repeat step 4 until there are no more values left in the sorted array or adding e_i to the interval will drop it below some purity threshold T.
    a. If there are no more values left in the sorted array, save the current interval.
    b. If the interval will drop below some purity threshold, remove all values equal to e_i from the current interval and save what is left. If possible, repeat step 2 with the same attribute but use a seed value greater than e_i.
7) For all saved intervals for attribute x_i, save the interval with the largest number of values.
8) Repeat step 2 with the next sorted array.
9) For all saved intervals for all attributes, save the interval with the largest number of values.
10) Using the saved interval from step 7, create a hyperblock.
11) Repeat step 1 with all n-D points not in a HB until all n-D points are within a hyperblock or no new more intervals can be made with any attribute.

## Merger Hyper

The second algorithm is Merger Hyperblock (MHyper). The idea for MHyper is that a hyperblock can be created by merging two overlapping hyperblocks.

The steps for the MHyper algorithm are as follows:

1) Seed an initial set of pure HBs with a single n-D point in each of them (HBs with length equal to 0).
2) Select a HB x from the set of all HBs.
3) Start iterating over the remaining HBs. If HBi has the same class as x then attempt to combine HBi with x to get a pure HB.
    a. Create a joint HB from HB_i and x that is an envelope around HB_i and x using the minimum and maximum of each attribute for HB_i and x.
    b. Check if any other n-D point y belongs to the envelop of HB_i and x. If y belongs to this envelope add y to the joint HB.
    c. If all points y in the joint HB are of the same class, then remove x and HB_i from the set of HBs that need to be changed.
4) Repeat step 3 for all remaining HBs that need to be changed. The result is a full pure HB that cannot be extended with other n-D points and continue to be pure.
5) Repeat step 2 for n-D points do not belong to already built full pure HBs.
6) Define an impurity threshold that limits the percentage of n-D points from opposite classes allowed in a dominant HB.
7) Select a HB x from the set of all HBs.
8) Attempt to combine x with remaining HBs.
    a. Create a joint HB from HB_i and x that is an envelope around HB_i and x.
    b. Check if any other n-D point y belongs to the envelop of HB_i and x. If y belongs to this envelope add y to the joint HB.
    c. Compute impurity of the HB_i (the percentage of n-D points from opposite classes introduced by the combination of x with HB_i.)
    d. Find HB_i with lowest impurity. If this lowest impurity is below predefined impurity threshold create a joint HB.
9) Repeat step 7 until all combinations are made.

## Interval Merger Hyper

The third algorithm is Interval Merger Hyper (IMHyper). The idea for IMHyper is to combine the IHyper and MHyper algorithms.

The steps for the IMHyper algorithm are as follows:

1) Run the IHyper algorithm.
2) Create a set of any n-D points not within the HBs created in step 1.
3) Run the MHyper algorithm on the set created in step 2 but add the HBs created in step 1 of this algorithm to the set of pure HBs created in step 1 of the MHyper algorithm.
