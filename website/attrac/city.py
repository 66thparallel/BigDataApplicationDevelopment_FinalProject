import os
from .attraction import Attraction

class City():
    def __init__(self, file):
        self.attractions = []
        city = file.split("/")[-1].split(".")[0]
        self.code = city
        if city == "nyc":
            city = "New York City"
            self.about = "Conquering New York in one visit is impossible. Instead, hit the must-sees – the Empire State Building, the Statue of Liberty, Central Park, the Metropolitan Museum of Art – and then explore off the beaten path with visits to The Cloisters or one of the city’s libraries. Indulge in the bohemian shops of the West Village or the fine dining of the Upper West Side. The bustling marketplace inside of Grand Central Station gives you a literal taste of the best the city has to offer."
            self.serialNum = "g60763"
        
        elif len(city) == 5:
            self.about = "Lingering over pain au chocolat in a sidewalk café, relaxing after a day of strolling along the Seine and marveling at icons like the Eiffel Tower and the Arc de Triomphe… the perfect Paris experience combines leisure and liveliness with enough time to savor both an exquisite meal and exhibits at the Louvre. Awaken your spirit at Notre Dame, bargain hunt at the Marché aux Puces de Montreuil or for goodies at the Marché Biologique Raspail, then cap it all off with a risqué show at the Moulin Rouge."
            self.serialNum = "g187147"
        else:
            self.about = "The crown jewels, Buckingham Palace, Camden Market…in London, history collides with art, fashion, food, and good British ale. A perfect day is different for everyone: culture aficionados shouldn't miss the Tate Modern and the Royal Opera House. If you love fashion, Oxford Street has shopping galore. For foodies, cream tea at Harrod’s or crispy fish from a proper chippy offers classic London flavor. Music and book buffs will love seeing Abbey Road and the Sherlock Holmes Museum (at 221B Baker Street, of course)."
            self.serialNum = "g186338"
        city = city[0].upper() + city[1:]
        self.name = city
        with open(file, 'r') as f:
            rank = 1
            result = 100
            for line in f.readlines():
                self.attractions.append(Attraction(rank, line, result))
                rank += 1
                result -= 5