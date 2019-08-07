# coding: utf-8
# !/usr/bin/python2
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: Profile the dataset.
"""

from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when, udf, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import size
import string


# Profile the dataset

# Load the dataset
citiesRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt")

## split the corpus using double pipes as delimiter
splitRDD = citiesRDD.map(lambda x: x.split("||"))

# Profile the data
col1 = splitRDD.map(lambda x: x[0])
col2 = splitRDD.map(lambda x: x[1])
col3 = splitRDD.map(lambda x: x[2])
col4 = splitRDD.map(lambda x: x[3])


## Examine what is in some of the columns
col2.take(10)
col4.take(10)

## convert rdd to dataframe
citiesDF = splitRDD.toDF(["username","rating","review","date"])

## count the total number of rows in the dataset
citiesDF.select("username").count()

## count the maximum number of characters in a user review
col3.map(lambda x: len(x)).max()

## get the types of all the columns
citiesDF.dtypes


# Clean and preprocess

## extract the needed columns (user rating and user review text)
extractDF = citiesDF.drop("username").drop("date")

## check user rating and review text are in the new dataframe
extractDF.show(10)

## use only reviews with rating of 10 or 50 for binary classification
citiesDF = extractDF.filter((extractDF.rating != '20') & (extractDF.rating != '30') & (extractDF.rating != '40'))

## check that only reviews with a rating of 10 or 50 are in the dataframe
citiesDF.show(30)