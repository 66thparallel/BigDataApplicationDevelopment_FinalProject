Final Project for CSCI-GA.3033 Big Data Application Development

Title: Tourist Attraction Sentiment Analysis with TripAdvisor Reviews

Authors: Yu-Ting Chiu, Jane Liu

Description: This application uses sentiment analysis to create a ranked list of the best offbeat tourist
    attractions for Paris, London, and NYC. This application can be used by travel industry professionals to identify emerging travel destinations. The application was written with Apache PySpark, the Spark MLlib machine learning library, and Python 2.7. The website and scraper were created with Python 3.6. It is unknown if the files will work correctly for other versions of Python.


# FILES
/home/[net ID]/bdad/fp/app_code/london_main.py
/home/[net ID]/bdad/fp/app_code/paris_main.py
/home/[net ID]/bdad/fp/app_code/nyc_main.py
/home/[net ID]/bdad/fp/data_ingest/scrape.s
/home/[net ID]/bdad/fp/data_ingest/scraper.py
/home/[net ID]/bdad/fp/etl_code/etl.txt
/home/[net ID]/bdad/fp/profiling_code/profile.py
/home/[net ID]/bdad/fp/website/attrac/static/london.jpg
/home/[net ID]/bdad/fp/website/attrac/static/paris.jpg
/home/[net ID]/bdad/fp/website/attrac/static/newyork.jpg
/home/[net ID]/bdad/fp/website/attrac/templates/base.html
/home/[net ID]/bdad/fp/website/attrac/templates/index.html
/home/[net ID]/bdad/fp/website/attrac/db.py
/home/[net ID]/bdad/fp/website/attrac/schema.sql
/home/[net ID]/bdad/fp/website/instance/flaskr.sqlite
/home/[net ID]/bdad/fp/website/output/london.txt
/home/[net ID]/bdad/fp/website/output/paris.txt
/home/[net ID]/bdad/fp/website/output/nyc.txt
/home/[net ID]/bdad/fp/website/hello.py
/home/[net ID]/bdad/fp/website/venv
/home/[net ID]/bdad/fp/screenshots/[screenshots of the running application]


# REQUIREMENTS
Please copy all files and folders into the working directory (shown above). The following libraries are used: PySpark, MLlib,
PySpark SQL, string, re.


# INSTRUCTIONS

1. A batch file scrape.s has been included to run the web scrapers on the Prince cluster. We used only one scraper.py file and updated the URLs to point to different tourist attractions for different cities.

    All input data is located in HDFS in bdad/fp/dataset/.

2. It is recommended to run the profiling code (profile.py) line by line in the REPL due to the verbose feedback from Spark  when running a script.

3. Please update the filepath variable with your local file path in london_main.py, paris_main.py, and nyc_main.py. It is located in the user-defined function printCity() at the bottom of the file.

    #### Set the file path for the output files
    `filepath = "/home/[net ID]/bdad/fp/website/output/"`

4. To run the executable files:
    ```
    cd /opt/cloudera/parcels/SPARK2/bin/
    ./spark2-submit --master local /home/[net ID]/bdad/fp/app_code/london_main.py
    ./spark2-submit --master local /home/[net ID]/bdad/fp/app_code/paris_main.py
    ./spark2-submit --master local /home/[net ID]/bdad/fp/app_code/nyc_main.py
    ```

5. The output text files (containing a ranked list of tourist attractions) is output to the file path specified
in #3 above. The files are:

    london.txt
    paris.txt
    nyc.txt

6. How to run the website:
    ```
    $ cd website
    $ . venv/bin/activate
    $ pip3 install flask
    $ export FLASK_APP=attrac
    $ export FLASK_ENV=development
    $ flask run   # server response should be "Running on http://127.0.0.1:5000/

    # You now can access the site on http://127.0.0.1:5000/
    ```
# EXPECTED RESULTS
The multinomial logistic regression model has approximately 30 - 67% accuracy depending on the training set.
