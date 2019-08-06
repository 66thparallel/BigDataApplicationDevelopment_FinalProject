# coding: utf-8
# !/usr/bin/python2
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: This application uses sentiment analysis to create a ranked list of the best tourist
    attractions for London, UK. It first obtains the data for each city from HDFS, merges it, tokenizes,
    removes stopwords, and lemmatizes. Because MLlib on Dumbo only supports binary logistic regression we use only
    those rows with rating=1 and rating=5 for binary classification. The data is then split into the training,
    validation, and test sets. We chose a combination of tf-idf and logistic regression as our model due to its
    popularity as an accurate technique for sentiment analysis. The data is trained, tested, and validated and
    the accuracy score is computed. We chose to use dataframes as the collection and MLib as the machine
    learning library.
"""
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import string

# Application for London, UK

# ETL

## Load the dataset
londonRDD = sc.textFile("bdad/fp/dataset/london/Buckingham_Palace.txt,bdad/fp/dataset/london/Camden_Market.txt,bdad/fp/dataset/london/Chelsea_FC_Stadium_Tour_Museum.txt,bdad/fp/dataset/london/Covent_Garden.txt,bdad/fp/dataset/london/Emirates_Stadium_Tour_and_Museum.txt,bdad/fp/dataset/london/Greenwich.txt,bdad/fp/dataset/london/HMS_Belfast.txt,bdad/fp/dataset/london/Highgate_Cemetery.txt,bdad/fp/dataset/london/Houses_of_Parliament.txt,bdad/fp/dataset/london/Imperial_War_Museums.txt,bdad/fp/dataset/london/Kensington_Gardens.txt,bdad/fp/dataset/london/Museum_of_London.txt,bdad/fp/dataset/london/Regent_s_Park.txt,bdad/fp/dataset/london/Shakespeare_s_Globe_Theatre.txt,bdad/fp/dataset/london/Sky_Garden.txt,bdad/fp/dataset/london/St_James_s_Park.txt,bdad/fp/dataset/london/St_Paul_s_Cathedral.txt,bdad/fp/dataset/london/The_View_from_The_Shard.txt,bdad/fp/dataset/london/Up_at_The_O2.txt,bdad/fp/dataset/london/Wallace_Collection.txt")

## Split the corpus using double pipes as delimiter
london_splitRDD = londonRDD.map(lambda x: x.split("||"))


# Clean and preprocess

## extract the needed columns (user rating and user review text)
temp1DF = london_splitRDD.toDF(["username","rating","review","date"])
temp2DF = temp1DF.drop("username").drop("date")
temp3DF = temp2DF.filter((temp2DF.rating != '20') & (temp2DF.rating != '30') & (temp2DF.rating != '40'))
londonDF = temp3DF.withColumn("rating", when(temp3DF["rating"] == "50", 1).otherwise(0))
londonDF.show(30)    # londonDF now has only reviews with rating of 1(label is 0) or 5(label is 1) for binary classification

## remove punctuation
london_punctDF = londonDF.where('review rlike "^[a-zA-Z]+"')

## tokenize, remove stopwords
london_tokenized = Tokenizer(inputCol="review", outputCol="tok_review").transform(london_punctDF)
london_tokenizedDF = london_tokenized.drop("review")
remover = StopWordsRemover(inputCol="tok_review", outputCol="stpwd_review")
london_stopwordsDF = remover.transform(london_tokenizedDF)

## lemmatize
london_finalDF = london_stopwordsDF


# Split data into training, validation, and test sets
(train_set, val_set, test_set) = london_finalDF.randomSplit([0.8, 0.1, 0.1], seed = 2000)


# Set up tf-idf
hashtf = HashingTF(numFeatures=2**12, inputCol="stpwd_review", outputCol='tf_review')
idf = IDF(inputCol='tf_review', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "rating", outputCol = "label")
pipeline = Pipeline(stages=[hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

# Train the binary classifier
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)

# Evaluate the accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print "Accuracy is", int(evaluator.evaluate(predictions)*100), "percent."


