#coding: utf-8
#!/usr/bin/python2.7
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: This application uses sentiment analysis to create a ranked list of the best offbeat tourist
    attractions in Paris, France. This predictive model can be used by travel industry professionals to 
    discover emerging travel destinations. For the training and validation data we used user reviews for 
    London and NYC. Next, we took data from tourist attractions ranked #11 - 30 on TripAdvisor and ran it 
    through our trained model to determine the best unusual/less well-known tourist attractions to visit 
    based on their sentiment score.
"""

from pyspark import SparkContext, SparkConf
from pyspark import sql
from pyspark.sql import DataFrame, SQLContext, Row
from pyspark.sql.functions import when, udf, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import string
import re

# Define Spark Context
conf = SparkConf().setAppName("finalProjectApp").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)

# ETL

# Load the dataset

trainRDD = sc.textFile("bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt,bdad/fp/dataset/nyc/Madison_Square_Garden.txt,bdad/fp/dataset/nyc/Manhattan_Skyline.txt,bdad/fp/dataset/nyc/Radio_City_Music_Hall.txt,bdad/fp/dataset/nyc/Rockefeller_Center.txt,bdad/fp/dataset/nyc/St_Patrick_s_Cathedral.txt,bdad/fp/dataset/nyc/Staten_Island_Ferry.txt,bdad/fp/dataset/nyc/The_Met_Cloisters.txt,bdad/fp/dataset/nyc/The_Museum_of_Modern_Art.txt,bdad/fp/dataset/nyc/The_Oculus.txt,bdad/fp/dataset/nyc/Times_Square.txt,bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/london/Kensington_Gardens.txt,bdad/fp/dataset/london/Museum_of_London.txt,bdad/fp/dataset/london/Regent_s_Park.txt,bdad/fp/dataset/london/Shakespeare_s_Globe_Theatre.txt,bdad/fp/dataset/london/Sky_Garden.txt,bdad/fp/dataset/london/St_James_s_Park.txt,bdad/fp/dataset/london/St_Paul_s_Cathedral.txt,bdad/fp/dataset/london/The_View_from_The_Shard.txt,bdad/fp/dataset/london/Up_at_The_O2.txt,bdad/fp/dataset/london/Wallace_Collection.txt")

# Split the corpus using double pipes as delimiter
splitRDD = trainRDD.map(lambda x: x.split("||"))

# Clean and preprocess

# profile the training set for skew. In the code below '10' stands for 1-star review, '20' is a 2-star review, etc.
count = []
temp1 = splitRDD.filter(lambda x: x[1]=='10')
temp2 = splitRDD.filter(lambda x: x[1]=='20')
temp3 = splitRDD.filter(lambda x: x[1]=='30')
temp4 = splitRDD.filter(lambda x: x[1]=='40')
temp5 = splitRDD.filter(lambda x: x[1]=='50')

# create a new training set with equal numbers of 1, 2, 3, 4, and 5-star reviews.
count.append((1, temp1.count()))
count.append((2, temp2.count()))
count.append((3, temp3.count()))
count.append((4, temp4.count()))
count.append((5, temp5.count()))

# in order to prevent skewed data find the maximum number of reviews for each star rating that can be used to train the model.
min_review = int(min(count,key=lambda x:x[1])[1])

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
# MLLib's multinomial logistic regression tool does not support RDDs
temp1DF = trainRDD.toDF(["username","rating","review","date"])
temp2DF = temp1DF.drop("username").drop("date")
trainDF = temp2DF

# NLP processing: tokenize, remove stopwords
tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(trainDF)
tokenizedDF = tokenized.drop("review")
remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
stopwordsDF = remover.transform(tokenizedDF)
finalDF = stopwordsDF

# Split data into training, validation, and test sets
(train_set, val_set) = finalDF.randomSplit([0.9, 0.1], seed = 2000)

# Set up tf-idf
tf = HashingTF(numFeatures=2**12, inputCol="stpwd_review", outputCol='tf_review')
idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "rating", outputCol = "label")
pipeline = Pipeline(stages=[tf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
## train_df.show(30)

# Train the classifier
lr = LogisticRegression(maxIter=100, labelCol="label", featuresCol="features")
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)
model_predict = predictions

# View the predictions of the validation set
## model_predict.select(model_predict.columns[:]).show(10)


# Apply the trained model to each tourist attraction to get its sentiment score
def getScore(rdd):

    # Preprocess and analyze the overall sentiment of a tourist attraction's user reviews
    splitRDD = rdd.map(lambda x: x.split("||"))
    puncRDD = splitRDD.map(lambda x: [x[0], x[1], re.sub('[^A-Za-z ]+', '', x[2].lower()), x[3]])
    temp1DF = puncRDD.toDF(["username", "rating", "review", "date"])
    temp2DF = temp1DF.drop("username").drop("date")
    tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(temp2DF)
    tokenizedDF = tokenized.drop("review")
    remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
    stopwordsDF = remover.transform(tokenizedDF)
    cityDF = stopwordsDF
    tf = HashingTF(numFeatures=2 ** 12, inputCol="stpwd_review", outputCol='tf_review')
    idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5)
    label_stringIdx = StringIndexer(inputCol="rating", outputCol="label")
    pipeline = Pipeline(stages=[tf, idf, label_stringIdx])
    pipelineFit = pipeline.fit(cityDF)
    results_df = pipelineFit.transform(cityDF)
    predictions = lrModel.transform(results_df)

    # display the predictions for this tourist attraction
    ## predictions.select(model_predict.columns[:]).show(10)

    # MLlib arbitrarily assigns labels to ratings each time. Find the label for each rating to calculate the sentiment score.
    star1df = results_df.filter(results_df["rating"]==10)
    star2df = results_df.filter(results_df["rating"]==20)
    star3df = results_df.filter(results_df["rating"]==30)
    star4df = results_df.filter(results_df["rating"]==40)
    star5df = results_df.filter(results_df["rating"]==50)

    label1 = results_df.where(results_df.rating==10).select("label").collect()[0][0] if bool(star1df.head(1))==True else 20.0
    label2 = results_df.where(results_df.rating==20).select("label").collect()[0][0] if bool(star2df.head(1))==True else 20.0
    label3 = results_df.where(results_df.rating==30).select("label").collect()[0][0] if bool(star3df.head(1))==True else 20.0
    label4 = results_df.where(results_df.rating==40).select("label").collect()[0][0] if bool(star4df.head(1))==True else 20.0
    label5 = results_df.where(results_df.rating==50).select("label").collect()[0][0] if bool(star5df.head(1))==True else 20.0

    # Create a sentiment score based on weighted averages for all labels
    num1 = predictions.where(predictions.label==label1).count()
    num2 = predictions.where(predictions.label==label2).count()
    num3 = predictions.where(predictions.label==label3).count()
    num4 = predictions.where(predictions.label==label4).count()
    num5 = predictions.where(predictions.label==label5).count()
    wavg1 = 1 * num1
    wavg2 = 2 * num2
    wavg3 = 3 * num3
    wavg4 = 4 * num4
    wavg5 = 5 * num5
    total = predictions.count()
    score = (wavg1 + wavg2 + wavg3 + wavg4 + wavg5)/float(total)

    return score

# Create a class to manage dataframes for each tourist attraction
class Paris:

    def __init__(self):
        self._paris_lum_RDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt")
        self._paris_basil_RDD = sc.textFile("bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt")
        self._paris_pere_RDD = sc.textFile("bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt")
        self._paris_laf_RDD = sc.textFile("bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt")
        self._paris_tuil_RDD = sc.textFile("bdad/fp/dataset/paris/Jardin_des_Tuileries.txt")
        self._paris_lat_RDD = sc.textFile("bdad/fp/dataset/paris/Latin_Quarter.txt")
        self._paris_mara_RDD = sc.textFile("bdad/fp/dataset/paris/Le_Marais.txt")
        self._paris_luxe_RDD = sc.textFile("bdad/fp/dataset/paris/Luxembourg_Gardens.txt")
        self._paris_jacq_RDD = sc.textFile("bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt")
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
        attractions = [(getScore(self._paris_lum_RDD), "Atelier_des_Lumieres"),(getScore(self._paris_basil_RDD), "Basilique_du_Sacre_Coeur_de_Montmartre"),(getScore(self._paris_pere_RDD), "Cemiterio_de_Pere_Lachaise"),(getScore(self._paris_laf_RDD), "Galeries_Lafayette_Paris_Haussmann"),(getScore(self._paris_tuil_RDD), "Jardin_des_Tuileries"),(getScore(self._paris_lat_RDD), "Latin_Quarter"),(getScore(self._paris_mara_RDD), "Le_Marais"),(getScore(self._paris_luxe_RDD), "Luxembourg_Gardens"),(getScore(self._paris_jacq_RDD), "Musee_Jacquemart_Andre"),(getScore(self._paris_marmot_RDD), "Musee_Marmottan_Monet"),(getScore(self._paris_rodin_RDD), "Musee_Rodin"),(getScore(self._paris_larmee_RDD), "Musee_de_l_Armee_des_Invalides"),(getScore(self._paris_obs_RDD), "Observatoire_Panoramique_de_la_Tour_Montparnasse"),(getScore(self._paris_panth_RDD), "Pantheon"),(getScore(self._paris_but_RDD), "Parc_des_Buttes_Chaumont"),(getScore(self._paris_vos_RDD), "Place_des_Vosges"),(getScore(self._paris_alex_RDD), "Pont_Alexandre_III"),(getScore(self._paris_germ_RDD), "Saint_Germain_des_Pres_Quarter"),(getScore(self._paris_notre_RDD), "Towers_of_Notre_Dame_Cathedral"),(getScore(self._paris_troc_RDD), "Trocadero")]
        return attractions

# Run the trained model on tourist attractions for each city
Paris = Paris()
paris_list = Paris.getList()
sorted_paris = sorted(paris_list, key=lambda x: x[0], reverse=True)

# Print the ranked list of tourist attractions for all cities
def printCity(list, name):

    # Set the file path for the output files
    filepath = "/home/jl860/bdad/fp/website/output/"

    filename = filepath + name + ".txt"
    with open(filename, 'w') as fp:
        fp.write('\n'.join('{}'.format(x[1]) for x in list))

printCity(sorted_paris, "paris")


# Calculate the accuracy by counting the number of predictions matching the label and dividing it by the total entries.
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print "\n\nAccuracy is %4.1f percent.\n\n" % (accuracy*100)
