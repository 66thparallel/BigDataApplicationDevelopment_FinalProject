# coding: utf-8
# !/usr/bin/python3
"""
Authors: Yu-Ting Chiu, Jane Liu
Description: A web scraper based on the Beautiful Soup 4 library. Scrapes the TripAdvisor Tourist
    Attractions pages for London, Paris, and NYC (London shown here). With thanks to Susan Li
    (https://towardsdatascience.com/@actsusanli).
"""

import requests
from bs4 import BeautifulSoup
import csv
import webbrowser
import io


def display(content, filename='output.html'):
    with open(filename, 'wb') as f:
        f.write(content)
        webbrowser.open(filename)


def get_soup(session, url, show=False):
    r = session.get(url)
    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200:  # not OK
        print('[get_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')


def post_soup(session, url, params, show=False):
    # read HTML from server and convert to Soup
    r = session.post(url, data=params)

    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200:  # not OK
        print('[post_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')


def scrape(url, lang='ALL'):
    # create session to keep all cookies, etc. between requests
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0',
    })

    items = parse(session, url)

    return items


def parse(session, url):
    # get number of reviews and start getting subpages with reviews
    print('[parse] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse] no soup:', url)
        return

    # num_reviews = soup.find('span', class_='reviews_header_count').text # get text
    num_reviews = soup.select(
        ".prw_rup.prw_filters_detail_language.ui_column.separated.is-3 > div > div.content > div.choices.is-shown-at-tablet > div:nth-child(2) > label > span.count")[
        0].text
    num_reviews = num_reviews[1:-1]
    num_reviews = num_reviews.replace(',', '')
    num_reviews = int(num_reviews)  # convert text into integer
    print('[parse] num_reviews ALL:', num_reviews)

    url_template = url.replace('Reviews-', 'Reviews-or{}-')
    print('[parse] url_template:', url_template)

    items = []

    offset = 0

    # Uncomment to scrape all reviews
    while(True):

        if offset > num_reviews:
            break
        subpage_url = url_template.format(offset)

        subpage_items = parse_reviews(session, subpage_url)
        if not subpage_items:
            break

        items += subpage_items

        if len(subpage_items) < 5:
            break

        offset += 10

    # for i in range(10):
    #     subpage_url = url_template.format(offset)

    #     subpage_items = parse_reviews(session, subpage_url)
    #     if not subpage_items:
    #         break

    #     items += subpage_items

    #     if len(subpage_items) < 5:
    #         break

    #     offset += 10
    return items


def get_reviews_ids(soup):
    items = soup.find_all('div', attrs={'data-reviewid': True})

    if items:
        reviews_ids = [x.attrs['data-reviewid'] for x in items][::2]
        print('[get_reviews_ids] data-reviewid:')
        return reviews_ids


def get_more(session, reviews_ids):
    url = 'https://www.tripadvisor.com/OverlayWidgetAjax?Mode=EXPANDED_HOTEL_REVIEWS_RESP&metaReferer=Hotel_Review'

    payload = {
        'reviews': ','.join(reviews_ids),  # ie. "577882734,577547902,577300887",
        # 'contextChoice': 'DETAIL_HR', # ???
        'widgetChoice': 'EXPANDED_HOTEL_REVIEW_HSX',  # ???
        'haveJses': 'earlyRequireDefine,amdearly,global_error,long_lived_global,apg-Hotel_Review,apg-Hotel_Review-in,bootstrap,desktop-rooms-guests-dust-en_US,responsive-calendar-templates-dust-en_US,taevents',
        'haveCsses': 'apg-Hotel_Review-in',
        'Action': 'install',
    }

    soup = post_soup(session, url, payload)

    return soup


def parse_reviews(session, url):
    # get all reviews from one page

    print('[parse_reviews] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    attraction_name = soup.find('h1', id='HEADING').text

    reviews_ids = get_reviews_ids(soup)
    if not reviews_ids:
        return

    soup = get_more(session, reviews_ids)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    items = []

    for idx, review in enumerate(soup.find_all('div', class_='reviewSelector')):

        # badgets = review.find_all('span', class_='badgetext')
        # if len(badgets) > 0:
        #     contributions = badgets[0].text
        # else:
        #     contributions = '0'

        # if len(badgets) > 1:
        #     helpful_vote = badgets[1].text
        # else:
        #     helpful_vote = '0'
        # user_loc = review.select_one('div.userLoc strong')
        # if user_loc:
        #     user_loc = user_loc.text
        # else:
        #     user_loc = ''

        user_name = review.select_one('div.info_text div')
        if user_name:
            user_name = user_name.text
        else:
            user_name = ''

        bubble_rating = review.select_one('span.ui_bubble_rating')['class']
        bubble_rating = bubble_rating[1].split('_')[-1]

        item = {
            'review_user_name': user_name + "||",
            'review_rating': bubble_rating + "||",
            'review_body': review.find('p', class_='partial_entry').text + "||",
            'review_date': review.find('span', class_='ratingDate')['title']  # 'ratingDate' instead of 'relativeDate'
        }

        items.append(item)
        # print('\n--- review ---\n')
        # for key,val in item.items():
        #     print(' ', key, ':', val)

    # print()

    return items


def write_in_csv(items, filename='results.csv',
                 headers=['hotel name', 'review title', 'review body',
                          'review date', 'contributions', 'helpful vote',
                          'user name', 'user location', 'rating'],
                 mode='w'):
    print('--- CSV ---')

    with io.open(filename, mode, encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, headers)

        # if mode == 'w':
        #     csv_file.writeheader()

        csv_file.writerows(items)


DB_COLUMN = 'review_body'
DB_COLUMN1 = 'review_date'
DB_COLUMN2 = 'review_rating'
DB_COLUMN3 = 'review_user_name'

##### Modify this section to scrape from a different city
prefix = "https://www.tripadvisor.com/Attraction_Review-g60763-"
suffix = "-New_York_City_New_York.html"
topElevenToThirty = ["d143358-Reviews-Intrepid_Sea_Air_Space_Museum",
                     "d103371-Reviews-Grand_Central_Terminal",
                     "d136053-Reviews-St_Patrick_s_Cathedral",
                     "d105123-Reviews-Rockefeller_Center",
                     "d267031-Reviews-Manhattan_Skyline",
                     "d110164-Reviews-Radio_City_Music_Hall",
                     "d143361-Reviews-Broadway",
                     "d106609-Reviews-The_Met_Cloisters",
                     "d136347-Reviews-Bryant_Park",
                     "d110145-Reviews-Times_Square",
                     "d104370-Reviews-Ellis_Island",
                     "d275371-Reviews-Madison_Square_Garden",
                     "d143367-Reviews-Greenwich_Village",
                     "d143363-Reviews-Staten_Island_Ferry",
                     "d10340693-Reviews-The_Oculus",
                     "d288031-Reviews-Chelsea_Market",
                     "d210108-Reviews-American_Museum_of_Natural_History",
                     "d12291624-Reviews-Gulliver_s_Gate",
                     "d105126-Reviews-The_Museum_of_Modern_Art",
                     "d10749293-Reviews-Christmas_Spectacular_Starring_the_Radio_City_Rockettes"
                     ]

topOneToTen = [
    "d105127-Reviews-Central_Park",
    "d1687489-Reviews-The_National_9_11_Memorial_Museum",
    "d105125-Reviews-The_Metropolitan_Museum_of_Art",
    "d104365-Reviews-Empire_State_Building",
    "d587661-Reviews-Top_of_the_Rock",
    "d103887-Reviews-Statue_of_Liberty",
    "d102741-Reviews-Brooklyn_Bridge",
    "d116236-Reviews-New_York_Public_Library",
    "d8072300-Reviews-One_World_Observatory",
    "_d267031-Reviews-Manhattan_Skyline",
]
#####

lang = 'en'

headers = [
    DB_COLUMN3,
    DB_COLUMN2,
    DB_COLUMN,
    DB_COLUMN1
]

for url in topOneToTen:

    # get all reviews for 'url' and 'lang'
    items = scrape(prefix + url + suffix, lang)

    # write in CSV
    filename = url.split('Reviews-')[1] + '__' + lang
    print('filename:', filename)
    # write_in_csv(items, filename + '.csv', headers, mode='w')
    with open("newyork/" + filename, 'w') as f:
        for item in items:
            temp = ""
            for v in item.values():
                temp += v
            f.write("%s\n" % temp)
