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

<p align="center"><img src="https://github.com/nancycyzl/ID3_from_scratch/blob/main/graph_log.PNG" width="30%"></p>

Then entropy (denoted as H) of the system is defined as the weighted sum of
log of probability of each possible value.

<p align="center">
<img src="https://latex.codecogs.com/svg.image?H(t)&space;=&space;\sum_{i=l}^{l}(P(t=i)\times&space;log_{2}(P(t=i)))" title="entropy" />
</p>

To calculate Information Gain, we can follows these steps:

1. Compute the entropy of original dataset wrt. target
2. For each descriptive feature, divide the dataset based on this feature's values
3. Sum entropy of each subset, which gives the remaining information. Note that D means dataset, d means descriptive features.

<p align="center">
<img src="https://latex.codecogs.com/svg.image?rem(d,D)&space;=&space;\sum_{l\in&space;level(d)}\frac{|D_{d=l}|}{D}&space;\times&space;H(t,D_{d=l})" title="rem" />
</p>

4. Subtracting remaining entropy from the original entropy gives the IG.

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;IG(d,G)&space;=&space;H(t,D)&space;-&space;rem(d,D)" title="IG" />
</p>

### ID3 Algorithm Explanation

After understanding the principal concept, Information Gain (IG),
let's get into how exactly a tree is expanded.

First, let's consider the original dataset. What is the first descriptive feature to split the dataset?
For every descriptive feature, we calculate the IG wrt. target feature, and choose the one that gives the largest IG.
The descriptive feature with the largest IG can best discriminate the level of target feature,
or in other words, make the system as pure as possible.

The whole dataset is now divided into multiple sub-datasets, each holding the same value of that descriptive feature.
For each subset, we repeat the steps to decide which feature to use to further split that subset.
One thing to note here is that for discrete descriptive features, a feature is only used once in a certain path.
In other words, after a dataset is splitted by feature A, we need drop A from our consideration when further spit its sub-datasets.

We need to know that the algorithm is a recursive algorithm, but the tree cannot expand infinitely.
When should we stop splitting a node? Here are the criteria.

* When all instances in a subset have the same target value (purest)
* When all descriptive features have been used before in this path
* When the subset is empty (no more instances to split)

### ID3 Algorithm pseudocode

> if all instances in D have the same target C:
>
>> return a tree consisting of a leaf node with label C
>>
>
> elif d is empty:
>
>> return a tree consisting of a leaf node with the label of the majoring target in C
>>
>
> elif D is empty:
>
>> return a tree consisting of a leaf node with the majoring label
>>
>
> else:
>
>> choose d[best] which gives the largest IG(d,D)  
>> make a new node, label is d[best]  
>> partition D using d[best]  
>> remove d[best] from d  
>> for each partition D_i of D:
>>> grow a branch by returning D = D_i
>>>
>>
