import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
from sklearn import linear_model
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# Some data structures that will be useful
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

#random.shuffle(allRatings)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

userIDs = {}
itemIDs = {}
idToUser = {}
idToItem = {}
for entry in allRatings:
    if(entry[0] not in userIDs):
        userIDs[entry[0]] = len(userIDs)
        idToUser[userIDs[entry[0]]] = entry[0]
    
    if(entry[1] not in itemIDs):
        itemIDs[entry[1]] = len(itemIDs)
        idToItem[itemIDs[entry[1]]] = entry[1]

meanRating = sum([r for _,_,r in ratingsTrain])/len(ratingsTrain)
learning_rate = 0.05
lamda1 = 0.00005 # 0.00005 and 0.00001 seem to be good with 0.00005 being better
lamda2 = 0.00005
lamda3 = 0.0001
lamda4 = 0.0001

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.5)

class RatingModel(tf.keras.Model):
    def __init__(self, meanRating, K, lamda1, lamda2, lamda3, lamda4):
        super(RatingModel, self).__init__()
        self.alpha = tf.Variable(meanRating)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.00001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.01))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.00001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.01))
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.lamda4 = lamda4
        #self.W = tf.Variable(tf.random.normal([], stddev=0.01))

    def reg(self):
        return self.lamda1 * tf.reduce_sum(self.betaU**2) + self.lamda2*tf.reduce_sum(self.betaI**2) + self.lamda3*tf.reduce_sum(self.gammaU**2) + self.lamda4*tf.reduce_sum(self.gammaI**2)# + self.lamda3*(self.W**2)
    
    def predict(self, user, item):
        #p = self.alpha + self.betaU[u] + self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        if(user not in userIDs and item not in itemIDs):
            gammaUAvg = tf.reduce_sum(self.gammaU, 0)/len(userIDs)
            #gammaUAvgMod = tf.sqrt(tf.reduce_sum(gammaUAvg**2))
            gammaIAvg = tf.reduce_sum(self.gammaI, 0)/len(itemIDs)
            #gammaIAvgMod = tf.sqrt(tf.reduce_sum(gammaIAvg**2))
            betaUAvg = tf.reduce_sum(self.betaU, 0)/len(userIDs)
            betaIAvg = tf.reduce_sum(self.betaI, 0)/len(itemIDs)
            p = self.alpha + betaUAvg + betaIAvg + tf.tensordot(gammaUAvg, gammaIAvg, 1)
            #p = self.alpha + betaUAvg + betaIAvg + self.W*(tf.tensordot(gammaUAvg, gammaIAvg, 1)/(gammaUAvgMod*gammaIAvgMod))
            return 5*tf.math.sigmoid(p)
        elif(user not in userIDs):
            i = itemIDs[item]
            gammaUAvg = tf.reduce_sum(self.gammaU, 0)/len(userIDs)
            #gammaUAvgMod = tf.sqrt(tf.reduce_sum(gammaUAvg**2))
            betaUAvg = tf.reduce_sum(self.betaU, 0)/len(userIDs)
            #gamma_i_mod = tf.sqrt(tf.reduce_sum(self.gammaI[i]**2))
            p = self.alpha + betaUAvg + self.betaI[i] + tf.tensordot(gammaUAvg, self.gammaI[i], 1)
            #p = self.alpha + betaUAvg + self.betaI[i] + self.W*(tf.tensordot(gammaUAvg, self.gammaI[i], 1)/(gammaUAvgMod*gamma_i_mod))
            return 5*tf.math.sigmoid(p)
        elif(item not in itemIDs):
            u = userIDs[user]
            gammaIAvg = tf.reduce_sum(self.gammaI, 0)/len(itemIDs)
            #gammaIAvgMod = tf.sqrt(tf.reduce_sum(gammaIAvg**2))
            betaIAvg = tf.reduce_sum(self.betaI, 0)/len(itemIDs)
            #gamma_u_mod = tf.sqrt(tf.reduce_sum(self.gammaU[u]**2))
            p = self.alpha + self.betaU[u] + betaIAvg + tf.tensordot(self.gammaU[u], gammaIAvg, 1)
            #p = self.alpha + self.betaU[u] + betaIAvg + self.W*(tf.tensordot(self.gammaU[u], gammaIAvg, 1)/(gamma_u_mod*gammaIAvgMod))
            return 5*tf.math.sigmoid(p)
        else:
            u = userIDs[user]
            i = itemIDs[item]
            #gamma_u_mod = tf.sqrt(tf.reduce_sum(self.gammaU[u]**2))
            #gamma_i_mod = tf.sqrt(tf.reduce_sum(self.gammaI[i]**2))
            p = self.alpha + self.betaU[u] + self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
            #p = self.alpha + self.betaU[u] + self.betaI[i] + self.W*(tf.tensordot(self.gammaU[u], self.gammaI[i], 1)/(gamma_i_mod*gamma_u_mod))
            return 5*tf.math.sigmoid(p)
        
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        #pred = self.alpha + beta_u + beta_i + self.W*(tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)/tf.multiply(tf.sqrt(tf.reduce_sum(gamma_u**2,1)),tf.sqrt(tf.reduce_sum(gamma_i**2,1))))
        return 5*tf.math.sigmoid(pred)
    
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return 2*tf.nn.l2_loss(pred - r) / len(sampleR)


def Train(model, trainData, epochs, numSamples, printLoss=True):
    sampleU, sampleI, sampleR = [], [], []
    for ind in range(0, len(trainData)):#numSamples):
        u,i,r = trainData[ind]#random.choice(trainData)
        sampleU.append(userIDs[u])
        sampleI.append(itemIDs[i])
        sampleR.append(r)

    vU, vI, vR = [], [], []
    for u,i,r in ratingsValid:
        vU.append(userIDs[u])
        vI.append(itemIDs[i])
        vR.append(r)

    for itr in range(epochs):
        with tf.GradientTape() as tape:
            # sampleU, sampleI, sampleR = [], [], []
            # for ind in range(0, len(trainData)):#numSamples):
            #     u,i,r = trainData[ind]#random.choice(trainData)
            #     sampleU.append(userIDs[u])
            #     sampleI.append(itemIDs[i])
            #     sampleR.append(r)

            loss = model(sampleU,sampleI,sampleR)
            loss += model.reg()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
        
        if (itr % 10 == 9 and printLoss):                 
            vMSE = model(vU,vI,vR).numpy()
            print("iteration " + str(itr+1) + ", objective = " + str(loss.numpy()) + ", validMSE = " + str(vMSE))

predictor = RatingModel(meanRating, 61, lamda1, lamda2, lamda3, lamda4)

Train(predictor, ratingsTrain, 100, 50000)

# MSE on validation set
sampleU, sampleI, sampleR = [], [], []
for u,i,r in ratingsValid:
    sampleU.append(userIDs[u])
    sampleI.append(itemIDs[i])
    sampleR.append(r)
    
validMSE = predictor(sampleU,sampleI,sampleR).numpy()
print(validMSE)

def roundFloat(a):
    if(a < 0):
        return 0.0

    if (a > 5):
        return 5.0

    # if(a - int(a) <= 0.1):
    #     return float(int(a))

    # if(a - int(a) >= 0.8):
    #     return float(int(a)+1)

    return a

predictions = open("predictions_Rating.csv", 'w')
    
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    
    # if(u not in userIDs or b not in itemIDs):
    #     predictions.write(u+","+b+","+str(roundFloat(meanRating))+"\n")
    # else:
    predictions.write(u+","+b+","+str(roundFloat(float(predictor.predict(u,b).numpy())))+"\n")
    
predictions.close()

