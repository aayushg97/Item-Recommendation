# Item-Recommendation

There are two tasks in this project :-

1. Read Prediction - Given a (user, book) pair, predict whether the user would read the book (0 or 1)
2. Rating Prediction - Given a (user, item) pair, predict the star rating that the user would give to the item (0-5)

## Read prediction

The approach uses the following two features :-

1. Whether the book is a popular book or not
2. Number of books read by the user that have a jaccard similarity (with the given book) greater than a certain threshold
	
These two features with 1/0 labels are used to train a logistic regression classifier. On the test set, the classifier is used to get classification scores for all (user, book) pairs and for every user, a list of books (mentioned with user in test set) is obtained. Half of these books that have higher scores are predicted as 1 and other half with lower scores is predicted as 0.

## Rating prediction

The approach uses a latent factor model which computes the rating as follows :-
		
rating = alpha + betaU + betaI + gammaU.gammaI
		
1. alpha &rarr; bias term
2. betaU &rarr; user specific parameter
3. betaI &rarr; item specific parameter
4. gammaU &rarr; latent features of user
5. gammaI &rarr; latent features of item		

All these parameters are learned for all (user, item) interactions. Regularizer of this model uses 2 lambdas, one for beta parameters and one for gamma parameters. Gamma vectors are of size K.

There is some handling for cold start cases as shown below.

1. User and item both are not seen :- 		rating = alpha
2. User is seen but item is not seen :- 	rating = alpha + betaU
3. User is not seen but item is seen :-		rating = alpha + betaI

Also, output of this model is clamped to be between 0 and 5.
