"""
Authors: Yu-Ting Chiu, Jane Liu
Description: main program of the websit
"""

import os
from .city import City
from flask import Flask, render_template


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/<city>')
    def attraction_city(city):
        filePath = "attrac/static/output/"
        n = len(city)
        if n==3:
            city = "nyc"
        filePath = filePath + city + ".txt"
        city = City(filePath)
        return render_template('attraction.html', city = city, url = "https://www.tripadvisor.com/Attraction_Review")

    @app.route('/')
    def index():
        return render_template('index.html')
    

    from . import db
    db.init_app(app)

    return app


