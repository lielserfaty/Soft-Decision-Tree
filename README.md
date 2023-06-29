# Soft-Decision-Tree

This repository contains the implementation of a Soft Decision Tree Classifier, which addresses a potential flaw in the traditional decision tree algorithm. In the decision tree algorithm, samples are routed through the tree based on binary conditions at each splitting node, leading to a deterministic prediction. However, this method fails to account for the uncertainty associated with decision tree splits.

## Problem Statement
The flaw in the traditional decision tree algorithm can be highlighted in the following scenarios:
1. When a sample is located close to the decision boundary.
2. When the split condition is based on a few training observations.


## Task 1: Soft Splits
To overcome the aforementioned flaw, we propose modifying the scikit-learn decision tree classifier to employ "soft splits" during inference. Soft splits allow each sample to be routed to either side of a split with a certain probability.

We will modify the decision tree classifier such that, during inference, each split will have an alpha probability (e.g., alpha = 10%) of being routed in the opposite direction of what the condition indicates and a 1-alpha probability (i.e., 1-alpha = 90%) of being routed according to the condition at the split node. To obtain the final prediction, the modified prediction algorithm will be run n times (e.g., n = 100) for each sample, and the probability vectors will be averaged.

Note:

- The code works for any number of target classes.
- Only the predict_proba function has been changed. The training procedure remains the same.

### Data Analysis
To evaluate the effectiveness of the algorithm, we analyzed its performance on 5 datasets, including a non-trivial classification dataset with more than 10 features and more than 1000 samples. 

### Model Evaluation
We evaluated the performance of the scikit-learn decision tree classifier and the modified implementation on the chosen dataset using accuracy and AUC as evaluation metrics. Repeated K-fold cross-validation has been used with 2 repetitions and 5 folds.

Additionally, we performed sensitivity analysis on the values of alpha and n.

## Task 2: Regression
We extend the modifications from Task 1 to address regression problems. Repeated all the steps mentioned earlier, adapting them to regression evaluation metrics such as MSE (Mean Squared Error).

## Task 3: Alternative Leaf Weighting
We proposed a different method to weigh all the leaves during the prediction of a decision tree classifier. The objective is to improve the performance compared to the method implemented in Task 1.
