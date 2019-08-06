# coding: utf-8
# !/usr/bin/python2
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: This application uses sentiment analysis to create a ranked list of the best tourist
    attractions for Paris, London, and NYC. It first obtains the data for each city from HDFS, merges it, tokenizes,
    removes stopwords, and lemmatizes. Because MLlib on Dumbo only supports binary logistic regression we use only
    those rows with rating=1 and rating=5 for two labels. The data is then split into the training, validation, and
    test sets. We chose a combination of tf-idf and logistic regression as our model due to its popularity as an
    accurate technique for sentiment analysis. The data is trained, tested, and validated and the accuracy score is
    computed. We chose to use dataframes as the collection and MLib as the machine learning library.
"""
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import string

'''
To run this application in PySpark REPL enter:
spark-submit --py-files main.py
'''

# Application for Paris, France

# Test code:
parisRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt")

# parisRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/paris/Musee_Marmottan_Monet.txt,bdad/fp/dataset/paris/Musee_Rodin.txt,bdad/fp/dataset/paris/Musee_de_l_Armee_des_Invalides.txt,bdad/fp/dataset/paris/Observatoire_Panoramique_de_la_Tour_Montparnasse.txt,bdad/fp/dataset/paris/Pantheon.txt,bdad/fp/dataset/paris/Parc_des_Buttes_Chaumont.txt,bdad/fp/dataset/paris/Place_des_Vosges.txt,bdad/fp/dataset/paris/Pont_Alexandre_III.txt,bdad/fp/dataset/paris/Saint_Germain_des_Pres_Quarter.txt,bdad/fp/dataset/paris/Towers_of_Notre_Dame_Cathedral.txt,bdad/fp/dataset/paris/Trocadero.txt")

# Split the corpus using double pipes as delimiter
paris_splitRDD = parisRDD.map(lambda x: x.split("||"))

# Clean and preprocess

## extract the needed columns (star rating and user review)
temp1DF = paris_splitRDD.toDF(["username","rating","review","date"])
temp2DF = temp1DF.drop("username").drop("date")
temp3DF = temp2DF.filter((temp2DF.rating != '20') & (temp2DF.rating != '30') & (temp2DF.rating != '40'))
parisDF = temp3DF.withColumn("rating", when(temp3DF["rating"] == "50", 1).otherwise(0))
parisDF.show(30)    # parisDF now has only reviews with rating of 1(label is 0) or 5(label is 1) for binary classification

## remove punctuation
paris_punctDF = parisDF.where('review rlike "^[a-zA-Z]+"')

## tokenize, remove stopwords
paris_tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(paris_punctDF)
paris_tokenizedDF = paris_tokenized.drop("review")
remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
paris_stopwordsDF = remover.transform(paris_tokenizedDF)

## lemmatize
paris_finalDF = paris_stopwordsDF

# Split data into training, validation, and test sets
(train_set, val_set, test_set) = paris_finalDF.randomSplit([0.98, 0.01, 0.01], seed = 2000)

# Set up tf-idf
hashtf = HashingTF(numFeatures=2**12, inputCol="stpwd_review", outputCol='tf_review')
idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "rating", outputCol = "label")
pipeline = Pipeline(stages=[hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

# Train the classifier
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)

# Evaluate the accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print "Accuracy is", int(evaluator.evaluate(predictions)*100), "percent."

# Application for London, UK

# londonRDD = sc.textFile("bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/london/Kensington_Gardens.txt,bdad/fp/dataset/london/Museum_of_London.txt,bdad/fp/dataset/london/Regent_s_Park.txt,bdad/fp/dataset/london/Shakespeare_s_Globe_Theatre.txt,bdad/fp/dataset/london/Sky_Garden.txt,bdad/fp/dataset/london/St_James_s_Park.txt,bdad/fp/dataset/london/St_Paul_s_Cathedral.txt,bdad/fp/dataset/london/The_View_from_The_Shard.txt,bdad/fp/dataset/london/Up_at_The_O2.txt,bdad/fp/dataset/london/Wallace_Collection.txt")



# Application for New York City, US

# nycRDD = sc.textFile("bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt,bdad/fp/dataset/nyc/Madison_Square_Garden.txt,bdad/fp/dataset/nyc/Manhattan_Skyline.txt,bdad/fp/dataset/nyc/Radio_City_Music_Hall.txt,bdad/fp/dataset/nyc/Rockefeller_Center.txt,bdad/fp/dataset/nyc/St_Patrick_s_Cathedral.txt,bdad/fp/dataset/nyc/Staten_Island_Ferry.txt,bdad/fp/dataset/nyc/The_Met_Cloisters.txt,bdad/fp/dataset/nyc/The_Museum_of_Modern_Art.txt,bdad/fp/dataset/nyc/The_Oculus.txt,bdad/fp/dataset/nyc/Times_Square.txt")











