class Attraction():
    def __init__(self, rank, name, result, serialNum = "d188757", rating = 35):
        self.ranking = rank
        self.title = name.replace("_", " ")
        self.about = "Home to Leonardo da Vinci's Mona Lisa, the Louvre\
                is considered the world's greatest art museum, with an unparalleled collection of items covering the full spectrum of art through the ages."
        self.serialNum = serialNum
        self.result = result
        self.rating = rating