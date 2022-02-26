ID3 From Scratch
----------------

Decision Tree is a very common Machine Learning model,
which can be used for both classification and regression tasks.
Commonly used algorithms for Decision Tree include ID3 and Cart.
This passage explains how ID3 algorithm works and
how to write an ID3 algorithm from scratch using python.

Note that only discrete descriptive features and discrete target are considered in this passage.
In other words, the algorithm developed in this passage is specifically for classification problem,
and we assume all descriptive features are discrete.
Dealing with continuous features or continuous target is a variation of this algorithm
which shall be discussed in another passage.

### ID3 explanation

Firstly we need to understand the structure of a tree in classification problems.

> Node: the descriptive feature that is used to split the tree
> Branch: the subclass that have the same value of the descriptive feature used to split the tree

For example, if we use "gender" to split a group of people,
the node will have value "gender", and the whole group is splitted into two:
one group contains all females and another contains all males.

However, there are usually many descriptive features in a dataset.
Which feature to choose to split a node is the core of tree expansion.
The rule is based on how well a descriptive feature discriminates between levels of target feature.
We use Information Gain (IG) to measure this ability.
Before understanding IG, there's a necessity to understand "Entropy".

#### Shannon's Entropy Model

> Entropy: measure how messy a system is

As entropy measures how messy a system is,
a more messy system has a higher entropy value.
In other words, a system which has a very low entropy value means that
the system is very "pure".
The goal of expanding a tree is that we want to make the leaves as pure as possible.
As we also know that, large probability of descriptive features means that
the system is relatively pure, or has a low entropy value
So we need to use a decreasing function to model probability of
descriptive features and entropy. A log function is able to do this.

<p align="center"><img src="pictures/graph_log.png" width="30%"></p>

Then entropy of the system is defined as the weighted sum of
log of probability of each possible value.

$$
H(t) = \sum_{j=l}^{l}(P(t=i)\times log_{2}(P(t=i)))

$$

To calculate Information Gain, we can follows these steps:

1. Compute the entropy of original dataset wrt. target
2. For each descriptive feature, divide the dataset based on this feature's values
3. Sum entropy of each subset, which gives the remaining information.

$$
rem(d,D) = \sum_{l\in level(d)}\frac{|D_{d=l}|}{D} \times H(t,D_{d=l})

$$

4. Subtracting remaining entropy from the original entropy gives the IG.

$$
IG(d,G) = H(t,D) - rem(d,D)

$$
