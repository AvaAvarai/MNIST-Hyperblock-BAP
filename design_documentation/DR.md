# Dimensionality Reduction

The DR method used is available in the publication: [Electronics 15(3), 580, Section 5.3](https://www.mdpi.com/2079-9292/15/3/580#sec5dot3-electronics-15-00580).

MNIST default dimensionality is 28 x 28 = 784-D. Crop 3 pixels all edges leaving us with 28 - 3 - 3 x 28 - 3 - 3 = 22 x 22 = 484-D. Average pooling with kernel of 2 x 2 with stride of 2 gives 22 / 2 x 22 / 2 = 11 x 11 = 121-D.

This give 121 / 784 = 0.15433673 = 15.4% of the original data.
