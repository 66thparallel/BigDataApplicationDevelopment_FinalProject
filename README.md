Final Project for CSCI-GA.3033 Big Data Application Development

Title: Tourist Attraction Sentiment Analysis with TripAdvisor Reviews

Authors: Yu-Ting Chiu, Jane Liu

Description: This application uses sentiment analysis to create a ranked list of the best offbeat tourist
    attractions for Paris, London, and NYC for travel industry professionals. This program was written in Python
    2.7. It is unknown if it will work correctly for other versions of Python.


FILES
app_code/main.py
data_ingest/scrape.s
data_ingest/scraper_london.py
data_ingest/scraper_paris.py
data_ingest/scraper_nyc.py
data_ingest/scraper_train.py
etl_code/etl.txt
profiling_code/profile.py
website/attrac/static/london.jpg
website/attrac/static/paris.jpg
website/attrac/static/newyork.jpg
website/attrac/templates/base.html
website/attrac/templates/index.html
website/attrac/db.py
website/attrac/schema.sql
website/instance/flaskr.sqlite
website/output/london.txt
website/output/paris.txt
website/output/nyc.txt
website/hello.py
screenshots/[screenshots of the running application]


REQUIREMENTS
Please copy all files and folders into the working directory (shown above). The following libraries are use: MLlib,
Spark SQL, string module.


INSTRUCTIONS

1. A batch file scrape.s has been included to run the web scrapers on the Prince cluster. The file paths and names
of the web scraper file should be updated.

2. It is recommended to run the profiling code (profile.py) in the REPL, due to the verbose feedback from Dubmo
server when running a PySpark script.

3. Please update the file path in main.py. It is this line in the user-defined function printCity() at the bottom of
the file.

    # Set the file path for the output files
    filepath = "/home/jl860/bdad/fp/website/output/"

4. To run the executable files: (please modify the file paths below)

cd /opt/cloudera/parcels/SPARK2/bin/
./spark2-submit --master local /home/jl860/bdad/fp/app_code/main.py


EXPECTED RESULTS
The binary logistic regression model is expected to have 80% accuracy.
We plan to train a multiple logistic regression model and will submit it in a separate file.