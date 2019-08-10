# coding: utf-8
# !/usr/bin/python2
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: Profile the dataset. Due to the output feedback from Dumbo, it's best to run these commands in REPL.
"""

from pyspark import SparkContext, SparkConf
from pyspark import sql
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when, udf, col
import string
import re


# Define Spark Context
conf = SparkConf().setAppName("finalProjectApp").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)

# Profile the dataset

# Load the dataset
citiesRDD = sc.textFile("bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/london/Kensington_Gardens.txt,bdad/fp/dataset/london/Museum_of_London.txt,bdad/fp/dataset/london/Regent_s_Park.txt,bdad/fp/dataset/london/Shakespeare_s_Globe_Theatre.txt,bdad/fp/dataset/london/Sky_Garden.txt,bdad/fp/dataset/london/St_James_s_Park.txt,bdad/fp/dataset/london/St_Paul_s_Cathedral.txt,bdad/fp/dataset/london/The_View_from_The_Shard.txt,bdad/fp/dataset/london/Up_at_The_O2.txt,bdad/fp/dataset/london/Wallace_Collection.txt,bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/paris/Musee_Marmottan_Monet.txt,bdad/fp/dataset/paris/Musee_Rodin.txt,bdad/fp/dataset/paris/Musee_de_l_Armee_des_Invalides.txt,bdad/fp/dataset/paris/Observatoire_Panoramique_de_la_Tour_Montparnasse.txt,bdad/fp/dataset/paris/Pantheon.txt,bdad/fp/dataset/paris/Parc_des_Buttes_Chaumont.txt,bdad/fp/dataset/paris/Place_des_Vosges.txt,bdad/fp/dataset/paris/Pont_Alexandre_III.txt,bdad/fp/dataset/paris/Saint_Germain_des_Pres_Quarter.txt,bdad/fp/dataset/paris/Towers_of_Notre_Dame_Cathedral.txt,bdad/fp/dataset/paris/Trocadero.txt,bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt,bdad/fp/dataset/nyc/Madison_Square_Garden.txt,bdad/fp/dataset/nyc/Manhattan_Skyline.txt,bdad/fp/dataset/nyc/Radio_City_Music_Hall.txt,bdad/fp/dataset/nyc/Rockefeller_Center.txt,bdad/fp/dataset/nyc/St_Patrick_s_Cathedral.txt,bdad/fp/dataset/nyc/Staten_Island_Ferry.txt,bdad/fp/dataset/nyc/The_Met_Cloisters.txt,bdad/fp/dataset/nyc/The_Museum_of_Modern_Art.txt,bdad/fp/dataset/nyc/The_Oculus.txt,bdad/fp/dataset/nyc/Times_Square.txt")

# split the corpus using double pipes as delimiter
splitRDD = citiesRDD.map(lambda x: x.split("||"))

# profile the data
col1 = splitRDD.map(lambda x: x[0])
col2 = splitRDD.map(lambda x: x[1])
col3 = splitRDD.map(lambda x: x[2])
col4 = splitRDD.map(lambda x: x[3])

# examine what is in the columns
col1.take(5)
col2.take(5)
col3.take(5)
col4.take(5)

# count the maximum number of characters in a user review
col3.map(lambda x: len(x)).max()

# check how skewed the data is
temp1 = splitRDD.filter(lambda x: x[1]=='10')
temp2 = splitRDD.filter(lambda x: x[1]=='20')
temp3 = splitRDD.filter(lambda x: x[1]=='30')
temp4 = splitRDD.filter(lambda x: x[1]=='40')
temp5 = splitRDD.filter(lambda x: x[1]=='50')
print "Number of 1-star user reviews: %d" % (temp1.count())
print "Number of 2-star user reviews: %d" % (temp2.count())
print "Number of 3-star user reviews: %d" % (temp3.count())
print "Number of 4-star user reviews: %d" % (temp4.count())
print "Number of 5-star user reviews: %d" % (temp5.count())

# create a new training set with equal numbers of 1, 2, 3, 4, and 5-star reviews.
count = []
count.append((1, temp1.count()))
count.append((2, temp2.count()))
count.append((3, temp3.count()))
count.append((4, temp4.count()))
count.append((5, temp5.count()))
min_review = int(min(count,key=lambda x:x[1])[1])  # find the smallest number of reviews available for a rating

# build new RDD with equal numbers of labeled reviews
train1RDD = sc.parallelize(temp1.take(min_review))
train2RDD = sc.parallelize(temp2.take(min_review))
train3RDD = sc.parallelize(temp3.take(min_review))
train4RDD = sc.parallelize(temp4.take(min_review))
train5RDD = sc.parallelize(temp5.take(min_review))
tempRDD = sc.union([train1RDD, train2RDD, train3RDD, train4RDD, train5RDD])

# remove punctuation and numbers
trainRDD = tempRDD.map(lambda x: [x[0], x[1], re.sub('[^A-Za-z ]+', '', x[2].lower()), x[3]])

# extract the needed columns (star rating and review text) and convert to dataframe
temp1DF = trainRDD.toDF(["username","rating","review","date"])
temp2DF = temp1DF.drop("username").drop("date")
citiesDF = temp2DF

# count the total number of rows in the dataset
citiesDF.select("rating").count()

# get the types of all the columns
citiesDF.dtypes