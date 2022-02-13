#coding: utf-8
#!/usr/bin/python2.7
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: This application uses sentiment analysis to create a ranked list of the best offbeat tourist
    attractions in London, UK. This predictive model can be used by travel industry professionals to discover 
    emerging travel destinations. For the training and test data we used user reviews for Paris and NYC. 
    Next, we took data from tourist attractions ranked #11 - 30 on TripAdvisor and ran it through our trained 
    model to determine the best unusual/less well-known tourist attractions to visit based on their sentiment score.
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

trainRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/paris/Musee_Marmottan_Monet.txt,bdad/fp/dataset/paris/Musee_Rodin.txt,bdad/fp/dataset/paris/Musee_de_l_Armee_des_Invalides.txt,bdad/fp/dataset/paris/Observatoire_Panoramique_de_la_Tour_Montparnasse.txt,bdad/fp/dataset/paris/Pantheon.txt,bdad/fp/dataset/paris/Parc_des_Buttes_Chaumont.txt,bdad/fp/dataset/paris/Place_des_Vosges.txt,bdad/fp/dataset/paris/Pont_Alexandre_III.txt,bdad/fp/dataset/paris/Saint_Germain_des_Pres_Quarter.txt,bdad/fp/dataset/paris/Towers_of_Notre_Dame_Cathedral.txt,bdad/fp/dataset/paris/Trocadero.txt,bdad/fp/dataset/nyc/American_Museum_of_Natural_History.txt,bdad/fp/dataset/nyc/Broadway.txt,bdad/fp/dataset/nyc/Bryant_Park.txt,bdad/fp/dataset/nyc/Chelsea_Market.txt,bdad/fp/dataset/nyc/Christmas_Spectacular_Starring_the_Radio_City_Rockettes.txt,bdad/fp/dataset/nyc/Ellis_Island.txt,bdad/fp/dataset/nyc/Grand_Central_Terminal.txt,bdad/fp/dataset/nyc/Greenwich_Village.txt,bdad/fp/dataset/nyc/Gulliver_s_Gate.txt,bdad/fp/dataset/nyc/Intrepid_Sea_Air_Space_Museum.txt,bdad/fp/dataset/nyc/Madison_Square_Garden.txt,bdad/fp/dataset/nyc/Manhattan_Skyline.txt,bdad/fp/dataset/nyc/Radio_City_Music_Hall.txt,bdad/fp/dataset/nyc/Rockefeller_Center.txt,bdad/fp/dataset/nyc/St_Patrick_s_Cathedral.txt,bdad/fp/dataset/nyc/Staten_Island_Ferry.txt,bdad/fp/dataset/nyc/The_Met_Cloisters.txt,bdad/fp/dataset/nyc/The_Museum_of_Modern_Art.txt,bdad/fp/dataset/nyc/The_Oculus.txt,bdad/fp/dataset/nyc/Times_Square.txt")

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

# Split data into training and test sets
(train_set, test_set) = finalDF.randomSplit([0.9, 0.1], seed = 2000)

# Set up tf-idf
tf = HashingTF(numFeatures=2**12, inputCol="stpwd_review", outputCol='tf_review')
idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "rating", outputCol = "label")
pipeline = Pipeline(stages=[tf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
test_df = pipelineFit.transform(test_set)
## train_df.show(30)

# Train the classifier
lr = LogisticRegression(maxIter=100, labelCol="label", featuresCol="features")
lrModel = lr.fit(train_df)
predictions = lrModel.transform(test_df)
model_predict = predictions

# View the predictions of the test set
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
class London:
    
    def __init__(self):
        self._lon_imp_RDD = sc.textFile("bdad/fp/dataset/london/Imperial_War_Museums.txt")
        self._lon_parl_RDD = sc.textFile("bdad/fp/dataset/london/Houses_of_Parliament.txt")
        self._lon_high_RDD = sc.textFile("bdad/fp/dataset/london/Highgate_Cemetery.txt")
        self._lon_belf_RDD = sc.textFile("bdad/fp/dataset/london/HMS_Belfast.txt")
        self._lon_green_RDD = sc.textFile("bdad/fp/dataset/london/Greenwich.txt")
        self._lon_emir_RDD = sc.textFile("bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt")
        self._lon_cov_RDD = sc.textFile("bdad/fp/dataset/london/Covent_Garden.txt")
        self._lon_chel_RDD = sc.textFile("bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt")
        self._lon_cam_RDD = sc.textFile("bdad/fp/dataset/london/Camden_Market.txt")
        self._lon_buck_RDD = sc.textFile("bdad/fp/dataset/london/Buckingham_Palace.txt")
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
        attractions = [(getScore(self._lon_imp_RDD), "Imperial_War_Museums"),(getScore(self._lon_parl_RDD), "Houses_of_Parliament"),(getScore(self._lon_high_RDD), "Highgate_Cemetery"),(getScore(self._lon_belf_RDD), "HMS_Belfast"),(getScore(self._lon_green_RDD), "Greenwich"),(getScore(self._lon_emir_RDD), "Emirates_Stadium_Tour_and_Museum"),(getScore(self._lon_cov_RDD), "Covent_Garden"),(getScore(self._lon_chel_RDD), "Chelsea_FC_Stadium_Tour_Museum"),(getScore(self._lon_cam_RDD), "Camden_Market"),(getScore(self._lon_buck_RDD), "Buckingham_Palace.txt"),(getScore(self._lon_kensing_RDD), "Kensington_Gardens"),(getScore(self._lon_lonmus_RDD), "Museum_of_London"),(getScore(self._lon_reg_RDD), "Regent_s_Park"),(getScore(self._lon_shakes_RDD), "Shakespeare_s_Globe_Theatre"),(getScore(self._lon_skygard_RDD), "Sky_Garden"),(getScore(self._lon_stjames_RDD), "St_James_s_Park"),(getScore(self._lon_stpaul_RDD), "St_Paul_s_Cathedral"),(getScore(self._lon_shard_RDD), "The_View_from_The_Shard"),(getScore(self._lon_02_RDD), "Up_at_The_O2"),(getScore(self._lon_wall_RDD), "Wallace_Collection")]
        return attractions


# Run the trained model on tourist attractions for each city
London = London()
london_list = London.getList()
sorted_london = sorted(london_list, key=lambda x: x[0], reverse=True)


# Print the ranked list of tourist attractions for all cities
def printCity(list, name):

    # Set the file path for the output files
    filepath = "/home/jl860/bdad/fp/website/output/"

    filename = filepath + name + ".txt"
    with open(filename, 'w') as fp:
        fp.write('\n'.join('{}'.format(x[1]) for x in list))

printCity(sorted_london, "london")

# Calculate the accuracy by counting the number of predictions matching the label and dividing it by the total entries.
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print "\nAccuracy is %4.1f percent.\n" % (accuracy*100)
