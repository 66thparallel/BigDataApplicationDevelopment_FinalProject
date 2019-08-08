# coding: utf-8
# !/usr/bin/python3
"""
Author: Yu-Ting Chiu, Jane Liu
Description: A web scraper utilizing the Beautiful Soup library. The scraper was used to retrieve user review
    data from the TripAdvisor Attractions page for London, Paris, and NYC.
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
    '''Read HTML from server and convert to Soup'''

    r = session.post(url, data=params)

    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200:  # not OK
        print('[post_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')


def scrape(url, lang='ALL'):
    # create session to keep all cookies (etc.) between requests
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0',
    })

    items = parse(session, url)

    return items


def parse(session, url):
    '''Get number of reviews and start getting subpages with reviews'''

    print('[parse] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse] no soup:', url)
        return

    # num_reviews = soup.find('span', class_='reviews_header_count').text # get text
    num_reviews = soup.select(
        ".prw_rup.prw_filters_detail_language.ui_column.separated.is-3 > div > div.content > div.choices.is-shown-at-tablet > div:nth-child(2) > label > span.count ")[
        0].text
    # taplc_detail_filters_ar_responsive_0 > div > div.collapsible.is-shown-at-tablet > div > div.ui_columns.filters > div
    num_reviews = num_reviews[1:-1]
    num_reviews = num_reviews.replace(',', '')
    num_reviews = int(num_reviews)  # convert text into integer
    print('[parse] num_reviews ALL:', num_reviews)

    url_template = url.replace('Reviews-', 'Reviews-or{}-')
    print('[parse] url_template:', url_template)

    items = []

    offset = 0

    # uncomment to scrap all reviews
    while (True):

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

    # scrape only 10 page
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
    '''Get all reviews from one page'''

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

##########modify this section to scrap from different city
prefix = "https://www.tripadvisor.com/Attraction_Review-g186338-"
suffix = "-London_England.html"
topElevenToThirty = ["d188159-Reviews-St_Paul_s_Cathedral",
                     "d548817-Reviews-Chelsea_FC_Stadium_Tour_Museum",
                     "d211708-Reviews-Houses_of_Parliament",
                     "d188126-Reviews-St_James_s_Park",
                     "d189047-Reviews-Covent_Garden",
                     "d187577-Reviews-Camden_Market",
                     "d187726-Reviews-Shakespeare_s_Globe_Theatre",
                     "d7398968-Reviews-Sky_Garden",
                     "d3539289-Reviews-The_View_from_The_Shard",
                     "d187601-Reviews-Greenwich",
                     "d3263267-Reviews-Up_at_The_O2",
                     "d187549-Reviews-Buckingham_Palace",
                     "d187675-Reviews-Regent_s_Park",
                     "d188893-Reviews-Kensington_Gardens",
                     "d187674-Reviews-Imperial_War_Museums",
                     "d1775127-Reviews-Emirates_Stadium_Tour_and_Museum",
                     "d187558-Reviews-Wallace_Collection",
                     "d187560-Reviews-Museum_of_London",
                     "d187551-Reviews-HMS_Belfast",
                     "d194290-Reviews-Highgate_Cemetery"
                     ]
############

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
    filename = url.split('Reviews-')[1] + '__' + "1-10"
    print('filename:', filename)
    # write_in_csv(items, filename + '.csv', headers, mode='w')
    with open(filename, 'w') as f:
        for item in items:
            temp = ""
            for v in item.values():
                temp += v
            f.write("%s\n" % temp)