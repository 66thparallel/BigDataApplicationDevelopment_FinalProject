# coding: utf-8
# !/usr/bin/python3
"""
Authors: Yu-Ting Chiu, Jane Liu
Classes:
    Main: Profile the dataset, merge, preprocess, and pass training data to the sentiment analysis model.

"""
import nltk
from preprocessor import *


# Load the data for all cities
parisRDD = sc.textFile("bdad/fp/dataset/paris/Atelier_des_Lumieres.txt,bdad/fp/dataset/paris/Basilique_du_Sacre_Coeur_de_Montmartre.txt,bdad/fp/dataset/paris/Cemiterio_de_Pere_Lachaise.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Galeries_Lafayette_Paris_Haussmann.txt,bdad/fp/dataset/paris/Jardin_des_Tuileries.txt,bdad/fp/dataset/paris/Latin_Quarter.txt,bdad/fp/dataset/paris/Le_Marais.txt,bdad/fp/dataset/paris/Luxembourg_Gardens.txt,bdad/fp/dataset/paris/Musee_Jacquemart_Andre.txt,bdad/fp/dataset/paris/Musee_Marmottan_Monet.txt,bdad/fp/dataset/paris/Musee_Rodin.txt,bdad/fp/dataset/paris/Musee_de_l_Armee_des_Invalides.txt,bdad/fp/dataset/paris/Observatoire_Panoramique_de_la_Tour_Montparnasse.txt,bdad/fp/dataset/paris/Pantheon.txt,bdad/fp/dataset/paris/Parc_des_Buttes_Chaumont.txt,bdad/fp/dataset/paris/Place_des_Vosges.txt,bdad/fp/dataset/paris/Pont_Alexandre_III.txt,bdad/fp/dataset/paris/Saint_Germain_des_Pres_Quarter.txt,bdad/fp/dataset/paris/Towers_of_Notre_Dame_Cathedral.txt,bdad/fp/dataset/paris/Trocadero.txt")

londonRDD = sc.textFile("")

nycRDD = sc.textFile("")

# Profile
londonRDD.take(10)
parisRDD.take(10)
nycRDD.take(10)

# split using double pipes as delimiter
splitRDD = parisRDD.map(lambda x: x.split("||"))

col1 = splitRDD.map(lambda x: x[0])  # column 1 contains user names
col2 = splitRDD.map(lambda x: x[1])  # column 2 contains star rating (1 - 5 stars)
col3 = splitRDD.map(lambda x: x[2])  # column 3 contains user review text
col4 = splitRDD.map(lambda x: x[3])  # column 4 contains review date

