# Hybrid-Similarity-for-Neighborhood-Based-CF

This project customizes Neighborhood-based Collaborative Filtering for Book-Crossing Dataset (BXDD) using a hybrid similarity measure.

The hybrid similarity measure is a combination of Bhattacharya measure and PIP(Proximity, Impact, Popularity) measure, which together works more efficiently than other conventional similarity measures when the dataset used is sparse, non-corated and also counters the infamous cold-start problem.

The conventional similarity measures have also been implemented in this project such as Pearson Coefficient, Cosine, Jaccard, Jaccard Mean Squared Difference and cJacMMD(Cosine, Jaccard, Mean Measure of Divergence) similarities. 

Numba Jit has been used in the code to provide optimized compilation in the absence of a GPU. 
