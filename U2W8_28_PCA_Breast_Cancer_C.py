
# Advanced Certification in AIML
## A Program by IIIT-H and TalentSprint

## Learning Objectives


At the end of the experiment, you will be able to:

* Apply PCA using sklearn package

#@title Experiment Walkthrough Video
from IPython.display import HTML
HTML("""<video width="854" height="480" controls>
<source src="https://cdn.iiith.talentsprint.com/aiml/Experiment_related_data/Walkthrough/PCA_Breast_Cancer_Walkthrough.webm" type="video/mp4">
</video>""")



## Dataset

### Description

skicit learn provides load_breast_cancer function to load and return the breast cancer wisconsin dataset (classification).

The breast cancer dataset is a classic and very easy binary classification dataset.

The dataset consists of 569 samples of 30 features with two classes as targets

## Importing required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

## Data Preparation

## Load the data from sklearn datasets

cancer = load_breast_cancer()

# Verify the features and targets names in load_breast_cancer dataset
print("Features of the data",cancer.data)
print("\nTargets of the data",cancer.target_names)

# Create a dataframe with cancer data
cancer_df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
cancer_df.shape

cancer_df.head()

cancer_df.describe()

# Extracting labels from cancer data
labels = cancer['target']
len(labels)

## Standardization (Scaling the data)

Standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis. For instance all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function.

So, for each observed value of the variable, subtract the mean and divide by the standard deviation.

Standardization of datasets is a common requirement for many machine learning estimators, which is implemented in scikit-learn

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cancer_df)
scaled_data

## Apply PCA on the scaled data

**Note:** Refer to the following [link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)



pca = PCA()
vectors = pca.fit_transform(scaled_data)

pca.explained_variance_ratio_ parameter returns a vector of the variance explained for each dimension.

PCA function provides explained_variance_ratio_ which gives the percentage of explained variance at each of the selected components.




variance = pca.explained_variance_ratio_
sorted(variance, reverse=True)

Cumulative variance in PCA gives the percentage of variance accounted for by the first n components.

For example, the cumulative percentage for the second component is the sum of the percentage of variance for the first and second component.

cumsum_explained_variance = np.cumsum(variance)
cumsum_explained_variance

Plotting the cumulative explained variance at each component

plt.plot(cumsum_explained_variance)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

## Plotting the variance percentage at each principal component

Cumulative sum is used to display the total sum of data as it grows with each component (or any other series or progression). It is view of the total contribution so far of a given measure against principal components

# Finding the variance between the each principal component
tot = sum(variance)
var_exp = [(i / tot)*100 for i in sorted(variance, reverse=True)]
plt.bar(range(1,len(var_exp)+1), var_exp)
plt.xlabel('Number of components')
plt.ylabel('Percentage of variance')
plt.show()

# Print the 'var_exp' and select the no of principal components where the highest variance is preserved
var_exp

The first 7 principal components together contain 91% of the information. So, remaining components can safely be dropped without losing too much information

## Apply PCA after selecting the principal components

pca = PCA(n_components=7)
reduced_data = pca.fit_transform(scaled_data)

reduced_data.shape

## Split the data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(reduced_data,labels)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

## Fit the model with reduced data

from sklearn.svm import SVC

model = SVC()
model.fit(X_train,Y_train)
acc = model.score(X_test, Y_test)
print("Accuracy of test data is",acc)

