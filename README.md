# Item-Recommendation

## Read prediction

The approach uses the following two features :-
	- whether the book is a popular book or not
	- number of books read by the user that have a jaccard similarity (with the given book) greater than a certain threshold
	
These two features with 1/0 labels are used to train a logistic regression classifier. On the test set, the classifier is used to get classification
scores for all user, book pairs and for every user, a list of books (mentioned with user in test set) is obtained. Half of these books that have higher 
scores are predicted as 1 and other half with lower scores is predicted as 0.

## Rating prediction

The approach uses a latent factor model with some modifications. The latent model in the lecture computes the following :-
		rating = alpha + betaU + betaI + gammaU.gammaI

In this computation, dot product of gammas is taking vector length of gammaU and gammaI in computation. But whether a user will like a book or not should
be decided only by the angle between these two vectors. So, cosine similarity between gammaU and gammaI might be a better way to model rating. This cosine
similarity can be scaled by a parameter W to train the model better. In short, the appraoch here computes the following :-
		rating = alpha + betaU + betaI + W x ((gammaU.gammaI)/(|gammaU||gammaI|))
		
Regularizer of this model uses 3 lambdas, one for beta parameters, one for gamma parameters and one for W. Gamma vectors are of size K=10

There is some handling for cold start cases as shown below.

user and item both are not seen :- 		rating = alpha
user is seen but item is not seen :- 	rating = alpha + betaU
user is not seen but item is seen :-	rating = alpha + betaI

Also, output of this model is clamped to be between 0 and 5
