import time
import numpy as np
import pyspark

from numpy import random
from pyspark import sql

sc = pyspark.SparkContext()

#Open rdd

rdd = sc.textFile("hdfs:///user/hadoop/recommend/ratings.csv")

#Remove header line
header = rdd.first()
rdd = rdd.filter(lambda x: x != header)

#Keep only (user, movie) information
rdd = rdd.map(lambda line : line.split(',')).map(lambda line : line[0] + ',' + line[2])

#Initialize number of classes and number of iterations
nb_z = 3
nb_iterations = 20

print("\n \n \n Starting PLSI Algorithm for "+str(nb_z)+" classes and "+str(nb_iterations)+" iterations \n \n \n")

#Compute the cartesian product of the (user, movie) couples with the 3 classes
classes = sc.parallelize(range(nb_z))

rdd = rdd.cartesian(classes)

rdd = rdd.distinct()

## Initialize q0 ##

#Order rdd by user, movie, class
ordered_rdd = rdd.map(lambda x: (x[0].split(','), x[1])).sortBy(lambda x : (x[0][0], x[0][1], x[1]))

#Create a vector of probabilities that sum to 1 every three probas
proba0 = np.random.rand(int(ordered_rdd.count()/nb_z), nb_z)
random_p = list((proba0 / np.reshape(proba0.sum(1), (int(ordered_rdd.count()/nb_z), 1))).flatten())

#Assign a probability to each triplet (user, movie, class)
q = ordered_rdd.map(lambda x : (x, random_p.pop(0)))
num_partitions = q.getNumPartitions()

#Create an empty list to keep track of the LogLikelihood
LogLik = []

###### Run the EM algorithm on nb_iterations #####

print("\n \n \n Preliminary work done! \n \n \n")

for i in range(nb_iterations) :

    start = time.time()
    
    #### M-STEP - Compute p(s|z) and p(z|u) based on q( z | (u,s) ) ####

    ## Compute p(s | z) : sum the probas associated to every couple (s,z) and divide it by the sum of probas associated to this z ##
    
    #Keep the probabilities of all the (movie, class) couples
    SZ_probas = q.map(lambda x: ((x[0][0][1], x[0][1]), x[1]))
    
    #Sum the probabilities for the same (movie, class) couples
    Nsz = SZ_probas.reduceByKey(lambda x,y: x + y)
    
    #Keep the probabilities associated to each class in the rdd and sum the probabilities by class
    Z_probas = q.map(lambda x: (x[0][1], x[1]))
    Nz = Z_probas.reduceByKey(lambda x,y: x+y)
    
    #Divide the probability of the (movie, class) couple by the probability of the class
    Nsz = Nsz.map(lambda x : (x[0][1], (x[0][0], x[1])))
    Psz = Nsz.join(Nz).coalesce(num_partitions)
    Psz = Psz.map(lambda x : ((x[1][0][0], x[0]), x[1][0][1] / x[1][1])) #This gives us p(s | u)
    
    ## Compute p(z | u) : sum the probas associated to every couple (u,z) and divide by the sum of probas associated to this u ##
    
    #Same idea : Keep the probabilities of all the (class, user) couples and sum them by couple
    ZU_probas = q.map(lambda x: ((x[0][0][0], x[0][1]), x[1]))
    Nzu = ZU_probas.reduceByKey(lambda x,y: x + y)
    
    # Keep the probabilities associated to each user in the rdd and sum the probabilities by user
    U_probas = q.map(lambda x: (x[0][0][0], x[1]))
    Nu = U_probas.reduceByKey(lambda x,y: x+y)
    
    #Divide the probability of the (class, user) couple by the probability of the user
    Nzu = Nzu.map(lambda x : (x[0][0], (x[0][1], x[1])))
    Pzu = Nzu.join(Nu).coalesce(num_partitions)
    Pzu = Pzu.map(lambda x : ((x[1][0][0], x[0]), x[1][0][1] / x[1][1])) #This gives us p(u | z)
    
    ### E-STEP - Compute new q( z | (u,s) ) = p(s|z)p(z|u) / sum ( p(s|z)p(z|u) )###
    
    ## For each (u,s,z), compute p(s | z) * p(z | u) ##
    
    #Here we want to join Pzu and Psz : to each triplet (u,s,z), we want to associate p(z|u) and p(s|z) (computed above)
    #We create couples (z,u) and (s,z) for each triplet (u,s,z) and change their places to make the join with Pzu and Psz possible
    
    q_int = q.map(lambda x : ((x[0][1], x[0][0][0]), (x[0][0][1], x[0][1])))
    q_int2 = q_int.join(Pzu).coalesce(num_partitions)
    q_int3 = q_int2.map(lambda x : (x[1][0], (x[0], x[1][1])))
    PzuPsz = q_int3.join(Psz).coalesce(num_partitions)
    
    #We now multiply p(z|u) and p(s|z) to obtain p(s|u)
    PzuPsz = PzuPsz.map(lambda x: ((x[1][0][0][1], x[0][0]), (x[0][1], x[1][0][1]*x[1][1])))
    
    ## For each (u,s), we compute sum ( p(s | z)* p(z | u) ) (summing over z) (this corresponds to p(s|u)) ##
    SumPzuPsz = PzuPsz.map(lambda x : (x[0], x[1][1])).reduceByKey(lambda x,y : x+y)
    
    #Update LogLikelihood
    log = SumPzuPsz.map(lambda x : np.log(x[1]))
    N = SumPzuPsz.count()
    L = log.reduce(lambda x,y : x+y)
    print("\n \n \n Iteration "+str(i+1)+ "complete, loglikelihood is: " + str(L/N)+"\n \n \n")
    LogLik.append(L/N)
    
    #For each (u,s,z), compute p(s|z)p(z|u) / sum( p(s|z)p(z|u) ) (this corresponds to the new q( z | (u,s) )
    q = PzuPsz.join(SumPzuPsz).coalesce(num_partitions)
    q = q.map(lambda x : ((x[0], x[1][0][0]), x[1][0][1]/x[1][1]))

    end = time.time()

    print("\n \n \n Iteration "+str(i+1)+" completed in "+str(end-start)+"\n \n \n")

#Build the recommendation

print("\n \n \n Building the recommendation \n \n \n ")

users = rdd.map(lambda x : x[0])
movies = rdd.map(lambda x : x[1])
classes = sc.parallelize(range(nb_z))
    
data = users.cartesian(movies)
data = data.cartesian(classes).map(lambda line : (line[0][0], line[0][1], line[1]))
data = data.distinct()

ordered_data = data.sortBy(lambda x : (x[0], x[1], x[2]))
couples = ordered_data.map(lambda x : ((x[2], x[0]), (x[1], x[2])))
probas = couples.join(Pzu).coalesce(num_partitions).map(lambda x : (x[1][0], (x[0], x[1][1])))
probas = probas.join(Psz).coalesce(num_partitions)
Psu = probas.map(lambda x : (x[1][0][0][1], x[0][0], x[1][0][1]*x[1][1]))
probs = Psu.map(lambda x : x[2])

threshold = np.quantile(probs.collect(), 0.9)

def prediction(rdd, threshold):
    return(rdd.map(lambda x : (x[0],x[1], x[2] >=threshold)))

result = prediction(Psu, threshold)

result.saveAsTextFile("hdfs:///user/hadoop/recommend/prediction")

sc.stop()
