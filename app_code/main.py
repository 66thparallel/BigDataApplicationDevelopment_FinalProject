# coding: utf-8
# !/usr/bin/python2
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: This application uses sentiment analysis to create a ranked list of the best offbeat tourist
    attractions for Paris, London, and NYC for travel industry professionals. For the training and testing data
    we chose the top 10 tourist attractions for each city on TripAdvisor. Then we took data from tourist attractions
    ranked #11 - 30 on TripAdvisor and ran it through our trained model to determine the best unusual/less well-known
    tourist attractions to visit based on their sentiment score.
"""

from pyspark import SparkContext, SparkConf
from pyspark import sql
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when, udf, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import string

# Define Spark Context
conf = SparkConf().setAppName("finalProjectApp").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)

# ETL

# Load the dataset

trainRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt")

# Split the corpus using double pipes as delimiter
splitRDD = trainRDD.map(lambda x: x.split("||"))

# Clean and preprocess

## extract the needed columns (user rating and user review text)
temp1DF = splitRDD.toDF(["username","rating","review","date"])
temp2DF = temp1DF.drop("username").drop("date")

## use only reviews with rating of 10 or 50 for binary classification
trainDF = temp2DF.filter((temp2DF.rating != '20') & (temp2DF.rating != '30') & (temp2DF.rating != '40'))
#trainDF.show(10)

# NLP processing: tokenize, remove stopwords
tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(trainDF)
tokenizedDF = tokenized.drop("review")
remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
stopwordsDF = remover.transform(tokenizedDF)
finalDF = stopwordsDF

# Split data into training, validation, and test sets
(train_set, val_set, test_set) = finalDF.randomSplit([0.8, 0.1, 0.1], seed = 2000)

# Set up tf-idf
tf = HashingTF(numFeatures=2**12, inputCol="stpwd_review", outputCol='tf_review')
idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "rating", outputCol = "label")
pipeline = Pipeline(stages=[tf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
#train_df.show(30)

# Train the classifier (label value 0.0 = positive, 1.0 = negative)
lr = LogisticRegression(maxIter=100, labelCol="label", featuresCol="features")
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)
model_predict = predictions

# Apply the trained model to each tourist attraction to get its sentiment score
def getScore(rdd):
    # Preprocess and analyze the overall sentiment of a tourist attraction's user reviews
    splitRDD = rdd.map(lambda x: x.split("||"))
    temp1DF = splitRDD.toDF(["username", "rating", "review", "date"])
    temp2DF = temp1DF.drop("username").drop("date").drop("rating")
    punctDF = temp2DF.where('review rlike "^[a-zA-Z]+"')
    tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(punctDF)
    tokenizedDF = tokenized.drop("review")
    remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
    stopwordsDF = remover.transform(tokenizedDF)
    cityDF = stopwordsDF
    tf = HashingTF(numFeatures=2 ** 12, inputCol="stpwd_review", outputCol='tf_review')
    idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5)
    pipeline = Pipeline(stages=[tf, idf])
    pipelineFit = pipeline.fit(cityDF)
    results_df = pipelineFit.transform(cityDF)
    predictions = lrModel.transform(results_df)
    num_positive = predictions.filter("prediction like '0.0'").count()
    total = predictions.count()
    score = num_positive/float(total)
    return score

# Create classes to manage dataframes for each tourist attraction
class London:

    def __init__(self):
        self._lon_kensing_RDD = sc.textFile("bdad/fp/dataset/london/Kensington_Gardens.txt")
        self._lon_lonmus_RDD= sc.textFile("bdad/fp/dataset/london/Museum_of_London.txt")
        self._lon_reg_RDD = sc.textFile("bdad/fp/dataset/london/Regent_s_Park.txt")
        self._lon_shakes_RDD = sc.textFile("bdad/fp/dataset/london/Shakespeare_s_Globe_Theatre.txt")
        self._lon_skygard_RDD= sc.textFile("bdad/fp/dataset/london/Sky_Garden.txt")
        self._lon_stjames_RDD = sc.textFile("bdad/fp/dataset/london/St_James_s_Park.txt")
        self._lon_stpaul_RDD = sc.textFile("bdad/fp/dataset/london/St_Paul_s_Cathedral.txt")
        self._lon_shard_RDD = sc.textFile("bdad/fp/dataset/london/The_View_from_The_Shard.txt")
        self._lon_02_RDD = sc.textFile("bdad/fp/dataset/london/Up_at_The_O2.txt")
        self._lon_wall_RDD = sc.textFile("bdad/fp/dataset/london/Wallace_Collection.txt")

    def getList(self):

        # Get a ranked list of less well-known tourist attractions based on sentiment
        attractions = [(getScore(self._lon_kensing_RDD), "Kensington_Gardens"),(getScore(self._lon_lonmus_RDD), "Museum_of_London"),(getScore(self._lon_reg_RDD), "Regent_s_Park"),(getScore(self._lon_shakes_RDD), "Shakespeare_s_Globe_Theatre"),(getScore(self._lon_skygard_RDD), "Sky_Garden"),(getScore(self._lon_stjames_RDD), "St_James_s_Park"),(getScore(self._lon_stpaul_RDD), "St_Paul_s_Cathedral"),(getScore(self._lon_shard_RDD), "The_View_from_The_Shard"),(getScore(self._lon_02_RDD), "Up_at_The_O2"),(getScore(self._lon_wall_RDD), "Wallace_Collection")]
        return attractions

class Paris:

    def __init__(self):
        self._paris_marmot_RDD = sc.textFile("bdad/fp/dataset/paris/Musee_Marmottan_Monet.txt")
        self._paris_rodin_RDD = sc.textFile("bdad/fp/dataset/paris/Musee_Rodin.txt")
        self._paris_larmee_RDD = sc.textFile("bdad/fp/dataset/paris/Musee_de_l_Armee_des_Invalides.txt")
        self._paris_obs_RDD = sc.textFile("bdad/fp/dataset/paris/Observatoire_Panoramique_de_la_Tour_Montparnasse.txt")
        self._paris_panth_RDD = sc.textFile("bdad/fp/dataset/paris/Pantheon.txt")
        self._paris_but_RDD = sc.textFile("bdad/fp/dataset/paris/Parc_des_Buttes_Chaumont.txt")
        self._paris_vos_RDD = sc.textFile("bdad/fp/dataset/paris/Place_des_Vosges.txt")
        self._paris_alex_RDD = sc.textFile("bdad/fp/dataset/paris/Pont_Alexandre_III.txt")
        self._paris_germ_RDD = sc.textFile("bdad/fp/dataset/paris/Saint_Germain_des_Pres_Quarter.txt")
        self._paris_notre_RDD = sc.textFile("bdad/fp/dataset/paris/Towers_of_Notre_Dame_Cathedral.txt")
        self._paris_troc_RDD = sc.textFile("bdad/fp/dataset/paris/Trocadero.txt")

    def getList(self):

        # Get a ranked list of less well-known tourist attractions based on sentiment
        attractions = [(getScore(self._paris_marmot_RDD), "Musee_Marmottan_Monet"),(getScore(self._paris_rodin_RDD), "Musee_Rodin"),(getScore(self._paris_larmee_RDD), "Musee_de_l_Armee_des_Invalides"),(getScore(self._paris_obs_RDD), "Observatoire_Panoramique_de_la_Tour_Montparnasse"),(getScore(self._paris_panth_RDD), "Pantheon"),(getScore(self._paris_but_RDD), "Parc_des_Buttes_Chaumont"),(getScore(self._paris_vos_RDD), "Place_des_Vosges"),(getScore(self._paris_alex_RDD), "Pont_Alexandre_III"),(getScore(self._paris_germ_RDD), "Saint_Germain_des_Pres_Quarter"),(getScore(self._paris_notre_RDD), "Towers_of_Notre_Dame_Cathedral"),(getScore(self._paris_troc_RDD), "Trocadero")]
        return attractions

class NYC:

    def __init__(self):
        self._nyc_madsn_RDD = sc.textFile("bdad/fp/dataset/nyc/Madison_Square_Garden.txt")
        self._nyc_skyl_RDD = sc.textFile("bdad/fp/dataset/nyc/Manhattan_Skyline.txt")
        self._nyc_radio_RDD = sc.textFile("bdad/fp/dataset/nyc/Radio_City_Music_Hall.txt")
        self._nyc_rock_RDD = sc.textFile("bdad/fp/dataset/nyc/Rockefeller_Center.txt")
        self._nyc_pat_RDD = sc.textFile("bdad/fp/dataset/nyc/St_Patrick_s_Cathedral.txt")
        self._nyc_stat_RDD = sc.textFile("bdad/fp/dataset/nyc/Staten_Island_Ferry.txt")
        self._nyc_cloi_RDD = sc.textFile("bdad/fp/dataset/nyc/The_Met_Cloisters.txt")
        self._nyc_mod_RDD = sc.textFile("bdad/fp/dataset/nyc/The_Museum_of_Modern_Art.txt")
        self._nyc_ocul_RDD = sc.textFile("bdad/fp/dataset/nyc/The_Oculus.txt")
        self._times_RDD = sc.textFile("bdad/fp/dataset/nyc/Times_Square.txt")

    def getList(self):

        # Get a ranked list of less well-known tourist attractions based on sentiment
        attractions = [(getScore(self._nyc_madsn_RDD), "Madison_Square_Garden"),(getScore(self._nyc_skyl_RDD), "Manhattan_Skyline"),(getScore(self._nyc_radio_RDD), "Radio_City_Music_Hall"),(getScore(self._nyc_rock_RDD), "Rockefeller_Center"),(getScore(self._nyc_pat_RDD), "St_Patrick_s_Cathedral"),(getScore(self._nyc_stat_RDD), "Staten_Island_Ferry"),(getScore(self._nyc_cloi_RDD), "The_Met_Cloisters"),(getScore(self._nyc_mod_RDD), "The_Museum_of_Modern_Art"),(getScore(self._nyc_ocul_RDD), "The_Oculus"),(getScore(self._times_RDD), "Times_Square")]
        return attractions

# Run the trained model on tourist attractions for each city
London = London()
london_list = London.getList()
sorted_london = sorted(london_list, key=lambda x: x[0])

Paris = Paris()
paris_list = Paris.getList()
sorted_paris = sorted(paris_list, key=lambda x: x[1])

NYC = NYC()
nyc_list = NYC.getList()
sorted_nyc = sorted(nyc_list, key=lambda x: x[1])


# Print the ranked list of tourist attractions for all cities
def printCity(list, name):

    # Set the file path for the output files
    filepath = "/home/jl860/bdad/fp/website/output/"

    filename = filepath + name + ".txt"
    with open(filename, 'w') as fp:
        fp.write('\n'.join('{}'.format(x[1]) for x in list))

printCity(sorted_london, "london")
printCity(sorted_paris, "paris")
printCity(sorted_nyc, "nyc")

# Spark doesn’t support accuracy as a metric for binary classification, so we calculated the accuracy by counting
# the number of predictions matching the label and dividing it by the total entries.

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
model_predict.select(model_predict.columns[:]).show(20)
print("Accuracy is", int(evaluator.evaluate(model_predict)*100), "percent.")
