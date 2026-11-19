
Course link: https://www.cs.toronto.edu/~rahulgk/courses/csc311_f24/index.html
# Introduction

+ **Supervised Learning**: have labeled examples of correct behavior
+ __Unsupervised Learning__: no labeled examples
+ __Reinforcement Learning__: learning system interacts with the world and learns to maximize a scalar reward signal.

**PyTorch, TensorFlow, JAX**
# Nearest Neighbour Methods

When we are focusing on _supervised learning_, we are given a **training set** consisting of inputs and corresponding labels, e.g.

![[Pasted image 20261116194004.png|420]]

1. To build a system that can use the provided training set in order to make predictions on new data.
2. Create mathematical notation for the problem.


e.g. Vector representation of an image
![[Pasted image 20261116194517.png|429]]


Training set
$$
D = \{ (x ^{1}, y^{1}), (x^{2}, y ^{2}), \dots (x ^{n} , y ^{n}) \}
$$

consists of a collection of pairs of an input vectors ${}x \in \mathbb{R} ^{d }{}$ and its corresponding _target_. 

tips:

superscript are used to indicate the index of the tuple in the training set.

Now we can start our optimization problem, the first ML algorithm: **Nearest Neighbour**.


Algorithm:

**Algorithm:** 1. Find example $(\mathbf{x}^{n}, y^{n})$ (from $\mathcal{D}_{tr}$) closest to $\mathbf{x}$. That is: $$\mathbf{x}^n = \operatorname*{argmin}_{i \in \{1, \dots, n\}} \text{distance}(\mathbf{x}^{(i)}, \mathbf{x})$$ 2. Output $y = y^{n}$.


Visualization: __Voronoi diagram__

Decision boundary: the boundary between regions of input space assigned to different categories

1-NN is sensitive when it comes to noise or mis-labeled data.

![[Pasted image 20261117084525.png|552]]

But we can handle it by having ${}k{}$ nearest neighbours vote.

![[Pasted image 20261117084601.png|477]]

Let ${}\mathcal{L}{}$ be the set of all class labels:

Algorithm (kNN):

1. Find ${}k {}$ examples ${}\{ (x^{1}, y ^{1}) \dots (x ^{n}, y ^{n}) \}{}$ in ${}\mathcal{D}_{tr}{}$ closest to the test instance ${}\mathbf{x}{}$
2. Classification output is majority class
$$

y ^{*} = \operatorname*{max}_{t ^{z} \in \mathcal{L}} \sum_{i = 1} ^{k } \mathbb{I}(t ^{z} = t ^{i})
$$

# Training, Validation and Test Sets.

Test sets are for computing test error, which can measure on unseen test data, for evaluating generalization performance.

Validation sets are for choosing [[hyperparameters]]. 

In general, we split the original dataset ${}\mathcal{D}{}$ into non-overlapping sets (60/30/10) ratio for train/valid/test

The test set is used only at the very end, to measure the generalization performance of the final configuration.

KNN - non parametric learning algorithm.

# Pitfalls: the curse of dimensionality

