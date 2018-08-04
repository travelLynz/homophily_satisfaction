import utils
import re
import os
import json
import settings as s
import pandas as pd
import utils
from nltk.tokenize import sent_tokenize
from googletrans import Translator

def translate_list(ls):
    translatedDic = {}
    for item in ls:
        # REINITIALIZE THE API
        translator = Translator()
        try:
            # translate the 'text' column
            if item is not None:
                translated = translator.translate(item, dest='en', src='it')
                translatedDic[item] = translated.text
        except Exception as e:
            print(str(e))
            continue
    return translatedDic

def format_verified(x):
    verified = []
    x = x.replace('[', '').replace(']', '').replace('"', '').replace('=','')
    if ("Guide" in x):
        verified.append("Guide")
        x = x.replace('Guide', '')
    x = re.sub(r'\([^()]*\)|([0-9]+)', '', x)
    v_list = list(filter(None,[s.strip() for s in x.split(',')]))
    return v_list+verified

def clean_guests(guests):
    print('Cleaning Guest Data')
    print('Initial number of Guest records:', len(guests))

    #Trim white spaces from column names
    print('Trimming white spaces from column names')
    guests = utils.trim_column_names(guests)

    # Cleaning wishlist column
    print('Cleaning up wishlist column')
    guests = guests.rename(columns={'whishListNumber;;;;;;;;;': 'wishListNumber'})
    guests = guests.rename(columns={'whishListNumber': 'wishListNumber'})
    guests['wishListNumber'] = [i.replace(';', '') if i is not None and type(i) != int else i for i in guests['wishListNumber']]
    guests['wishListNumber'] = [i.replace(',', '') if i is not None and type(i) != int else i for i in guests['wishListNumber']]

    #
    print('Removing records with NULL ids:', len(guests[guests['id'] == 'null']), ' records')
    guests = guests[guests['id'] != 'null']
    print('Updated number of guest records:', len(guests))

    #Drop duplicates
    print("Drop duplicate records")
    guests = guests.drop_duplicates()
    print('Updated number of guest records:', len(guests))

    #Clean up quotation marks
    print('Cleaning up quotations')
    guests['wishListNumber'] = guests['wishListNumber'].apply(utils.clean_quotations)
    guests['id'] = guests['id'].apply(utils.clean_quotations)

    #Change Guide Number to integer
    print('Changing guideNumber to integer')
    guests['guideNumber'] = [i.replace(',', '') if i is not None and type(i) != int else i for i in guests['guideNumber']]
    guests['guideNumber'] = [int(i) if i is not None and type(i) != int else i for i in guests['guideNumber']]

    #Remove records where guests have no membershipDate
    print('Removing records where guests have no membershipDate', len(guests[(guests['membershipDate'] == 'null') | (guests['membershipDate'].isnull())]), ' records')
    guests = guests[(guests['membershipDate'] != 'null') & (~guests['membershipDate'].isnull())]
    print('Updated number of guest records:', len(guests))

    print('Split membershipDate into membershipMonth and membershipYear')
    guests[['membershipMonth','membershipYear']] = guests['membershipDate'].str.split(' ', expand=True,)
    guests = guests.drop('membershipDate', axis=1)

    #
    print('Removing records with non-numeric ids:')
    guests = guests[guests.id.apply(lambda x: x.isnumeric())]
    print('Updated number of guest records:', len(guests))

    return guests

def clean_reviews(reviews):
    print('Cleaning Review Data')
    print('Initial number of Review records:', len(reviews))

    #Trim white spaces from column names
    print('Trimming white spaces ')
    reviews = reviews.applymap(lambda x: x.strip() if type(x) is str else str(x))

    # Remove Line Breaks
    print('Cleaning up Line Breaks')
    reviews['comments'] = reviews['comments'].map(lambda x: x.replace("\r\n", " ").replace("\n", " ").replace("\r", " "))

    # Remove Line Breaks
    print('Adding cancellation marker')
    reviews['hostCancelled'] = reviews['comments'].map(lambda x: 'Y' if (x.startswith('The host cancelled') or x.startswith('The host canceled') or x.startswith('The reservation was cancelled') or x.startswith('The reservation was canceled')) else 'N')

    return reviews

def trans_verified_list(ls, veri_dict):

    return [veri_dict[i] for i in ls]

def translate_guests(guests):

    print("Translating months")
    guests['membershipMonth'] = guests.membershipMonth.map(lambda x: s.months_translated[x] if x in s.months_translated.keys() else x)

    guests['linkedAccountVerified'] = guests['linkedAccountVerified'].apply(lambda x: format_verified(x) if x != None else [])
    veri_set = set()
    for i in guests['linkedAccountVerified']:
        veri_set |= set(i)

    print("Distinct number of verifications: ", len(veri_set))
    print("Translating verifications")
    if os.path.exists('trans_verified.json'):
        old_translations = json.load(open('trans_verified.json'))
        new_translations = list(veri_set - old_translations.keys())
        print("Number of new translations needed for verifications: ", len(new_translations))
        trans_verified = utils.merge_two_dicts(old_translations, translate_list(new_translations)) if len(new_translations) > 0 else old_translations
    else:
        u_verified = list(veri_set)
        trans_verified = translate_list(u_verified)
    utils.save_dict_as_json(trans_verified, 'trans_verified')
    guests['linkedAccountVerified'] = guests.linkedAccountVerified.map(lambda x: str(trans_verified_list(x, trans_verified)) if x != None else None)

    u_cities = guests['city'].unique()
    print("Distinct number of cities: ", len(u_cities))
    print("Translating cities")
    if os.path.exists('trans_cities.json'):
        old_translations = json.load(open('trans_cities.json'))
        new_translations = list(set(u_cities) - old_translations.keys())
        print("Number of new translations needed for cities: ", len(new_translations))
        trans_cities = utils.merge_two_dicts(old_translations, translate_list(new_translations)) if len(new_translations) > 0 else old_translations
    else:
        trans_cities = translate_list(u_cities)
    utils.save_dict_as_json(trans_cities, 'trans_cities')
    guests['city'] = translate_cities(guests, 'city')

    return guests

def translate_cities(tbl, col):
    u_cities = tbl[col].unique()
    print("Distinct number of cities: ", len(u_cities))
    print("Translating cities")
    if os.path.exists('trans_cities.json'):
        old_translations = json.load(open('trans_cities.json'))
        new_translations = list(set(u_cities) - old_translations.keys())
        print("Number of new translations needed for cities: ", len(new_translations))
        trans_cities = utils.merge_two_dicts(old_translations, translate_list(new_translations)) if len(new_translations) > 0 else old_translations
    else:
        trans_cities = translate_list(u_cities)
    utils.save_dict_as_json(trans_cities, 'trans_cities')
    return tbl[col].map(lambda x: trans_cities[x] if x != None else x)

def get_listing_info(raw):
    print("Retrieving %d columns for listings" % len(s.listing_cols))
    listings = raw[s.listing_cols]
    listings = listings.drop_duplicates(subset=None, inplace=False)
    print("Number of NYC listings: %d" % len(listings))
    return listings

def get_host_info(raw):
    print("Retrieving %d columns for hosts" % len(s.host_cols))
    hosts = raw[s.host_cols]
    hosts = hosts.drop_duplicates(subset=None, inplace=False)
    print("Number of NYC hosts: %d" % len(hosts))

    #Remove host_ from column names
    print("Removing \'host_\' from column names")
    hosts = hosts.rename(columns=lambda x: x.replace('host_', ''))

    return hosts

def get_review_info(raw, listings):
    print("Retrieving %d columns for reviews" % len(s.review_cols))
    reviews = raw[s.review_cols]
    print("Number of NYC reviews: %d" % len(reviews))

    #Retrieving hostID from listing info"
    print("Retrieving hostID from listing info")
    reviews['recipient_id'] = reviews.join(listings.set_index('id'), on='listing_id')['host_id']

    return reviews

def restrict_multiple_listings(hosts, listings, reviews, guests):

    # Reduced Hosts
    new_hosts = hosts[(hosts['calculated_listings_count'] == 1) & (hosts['listings_count'] == 1)]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['host_id'].isin(set(new_hosts['id'].astype(str)))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Reviews
    new_reviews = reviews[reviews['recipient_id'].isin(set(new_hosts['id'].astype(str)))]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(new_overall_guests)]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)

def restrict_number_of_reviews(hosts, listings, reviews, guests):

    # Reduced Hosts
    host_review_count = reviews.groupby('recipient_id').count()[['id']].rename(columns={'id': 'num_of_reviews'})
    new_host_ids = set([str(i) for i in host_review_count[host_review_count['num_of_reviews'] >4].index])

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(new_host_ids)]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['host_id'].isin(new_host_ids)]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Reviews
    new_reviews = reviews[reviews['recipient_id'].isin(new_host_ids)]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(set(utils.convert_to_str(new_overall_guests)))]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)


def remove_no_reviews(hosts, listings, reviews):

    # Remove Listings with no reviews
    print('Remove Listings with no reviews')
    listings_with_reviews = reviews['listing_id'].unique()
    new_listings = listings[listings['id'].isin(listings_with_reviews.astype(str))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Remove Hosts with no reviews
    print('Remove Hosts with no reviews')
    hosts_with_reviews = reviews['recipient_id'].unique()
    new_hosts = hosts[hosts['id'].isin(hosts_with_reviews.astype(str))]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    return (new_hosts, new_listings)

def add_cancellation_col(hosts, reviews):
    h = hosts.copy()
    cancellation_table = reviews[reviews['hostCancelled'] == 'Y'].groupby(['recipient_id']).size().reset_index(name='num_of_cancellations')
    h['id'] = h['id'].astype(str)
    h = h.join(cancellation_table.set_index('recipient_id'), on='id')
    h['num_of_cancellations'] = h['num_of_cancellations'].fillna(0)
    return h

def add_reviews_length_cols(reviews):
    reviews['token_len'] = [int(len(utils.tokenize(c))) for c in reviews['comments']]
    reviews['num_of_sents'] = [int(len(sent_tokenize(c))) for c in reviews['comments']]
    return reviews

def remove_cancellations(hosts, listings, reviews, guests):

    # Remove Cancellations
    print('Removing cancellation notifications from reviews')
    new_reviews = reviews[reviews['hostCancelled'] == 'N']
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(set(new_reviews['recipient_id'].astype(str)))]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['id'].isin(set(new_reviews['listing_id'].astype(str)))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(new_overall_guests)]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)

def restrict_review_length(hosts, listings, reviews, guests):

    # Reduced Reviews 1
    print('Removing empty('', None, nan) reviews')
    nempty_reviews = reviews[(reviews['comments'] != 'None') & (reviews['comments'] != '') & (reviews['comments'] != 'nan') & (~reviews['comments'].isnull())]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(nempty_reviews), utils.get_decreased_percent(nempty_reviews, reviews)))

    # Reduced Reviews 2
    print('Removing reviews less than 5 words')
    new_reviews = nempty_reviews[nempty_reviews['token_len'] >= 5]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, nempty_reviews)))

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(set(new_reviews['recipient_id'].astype(str)))]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['id'].isin(set(new_reviews['listing_id'].astype(str)))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(new_overall_guests.astype(str))]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)

def restrict_review_langs(hosts, listings, reviews, guests):

    # Reduced Reviews langs
    print('Removing reviews based on Language Restrictions')

    agg_restrict = reviews[(reviews['google_langs'] == reviews['langdetect_langs']) & (reviews['google_langs'] == 'en') & (reviews['google_langs_conf'] > 0.9) & (reviews['langdetect_langs_conf'] > 0.9)]
    print("-Reviews that have 'English' language detection agreements between 'langdetect' and googletrans = %d" % len(agg_restrict))

    unk_restrict = reviews[(reviews['google_langs'] == 'unk') & (reviews['langdetect_langs'] == 'en') & (reviews['langdetect_langs_conf'] >= 0.99)]
    print("-Reviews that contained emoticons in them preventing 'googletrans' from correct detection = %d" % len(unk_restrict))

    new_reviews = pd.concat([agg_restrict, unk_restrict]).drop_duplicates(subset=None, inplace=False)
    print("-Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(set(new_reviews['recipient_id'].astype(str)))]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['id'].isin(set(new_reviews['listing_id'].astype(str)))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(new_overall_guests.astype(str))]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)

def show_reviews_per_person_dist(reviews_counts, idcol):
    count_table = utils.create_value_counts_table(reviews_counts, 'num_of_reviews', 'num_of_guest_reviews')
    print(count_table.head(10))
    utils.print_summary(reviews_counts['num_of_reviews'], False)

def build_exclusion_table(review_counts, reviews, listings, low, high):
    table = pd.DataFrame({"Exclusion(By # of reviews per host)": ["Hosts", "Reviews", "Guests", "Listings"]})
    table["Pre-exclusion"] = [len(review_counts), len(reviews), len(reviews['reviewer_id'].unique()), len(listings)]
    for i in range(low, high):
        ids = review_counts[review_counts['num_of_reviews'] > i].index
        new_reviews = reviews[reviews['recipient_id'].isin(ids)]
        new_listings = listings[listings['host_id'].isin(ids)]
        table["> " + str(i)] = [str(len(ids)) + " (" + str(format(len(ids)*100/table["Pre-exclusion"][0], ".2f")) + "%)",
                         str(len(new_reviews)) + " (" + str(format(len(new_reviews)*100/table["Pre-exclusion"][1], ".2f")) + "%)",
                         str(len(new_reviews['reviewer_id'].unique())) + " (" + str(format(len(new_reviews['reviewer_id'].unique())*100/table["Pre-exclusion"][2], ".2f")) + "%)",
                         str(len(new_listings)) + " (" + str(format(len(new_listings)*100/table["Pre-exclusion"][3], ".2f")) + "%)"]
    return table

def isLocation(x):
    res = x.strip()
    if utils.hasNumbers(res) or res == "" or res.replace(",", "").strip() == "":
        return False
    return True

def clean_hosts_hosts(x):
    result = re.sub(r'[\(\)]|[\\\[\\\]]','',x.replace("), (Da ", "(").replace("(Da ", "("))
    result = [i.strip() for i in result.split("·") if isLocation(i)]
    return result

def create_visitation_table(tbl):
    city_matches = [[r['hostId '], v] for i, r in tbl.iterrows() for v in r['hostedBy']]
    result = pd.DataFrame(data=city_matches, columns=["hostId", "visited"])
    result[["hostId", "visited", "min_times"]] = result.groupby(["hostId", "visited"]).size().reset_index()
    return result[~result['hostId'].isnull()]

def get_host_visitation_table(host_hosts):
    host_hosts.is_copy = False
    host_hosts['hostedBy'] = host_hosts[' hostedBy'].map(lambda x: clean_hosts_hosts(x))
    return create_visitation_table(host_hosts)

def clean_host_reviews(reviews):
    print('Cleaning Review Data')
    print('Initial number of Review records:', len(reviews))

    #Trim white spaces from column names
    print('Trimming white spaces from column names')
    reviews = utils.trim_column_names(reviews)

    #Trim white spaces from column names
    print('Trimming white spaces ')
    reviews = reviews.applymap(lambda x: x.strip() if type(x) is str else str(x))

    # Remove Line Breaks
    print('Cleaning up Line Breaks')
    reviews['comments'] = reviews['comments'].map(lambda x: x.replace("\r\n", " ").replace("\n", " ").replace("\r", " "))

    #
    print('Removing records with NULL guest ids:', len(reviews[(reviews['guestId'] == 'null') & (reviews['guestId'].isnull())]), ' records')
    reviews = reviews[(reviews['guestId'] != 'null') & (~reviews['guestId'].isnull())]
    print('Updated number of host review records:', len(reviews))

    #
    print('Removing records with NULL host ids:', len(reviews[(reviews['hostId'] == 'null') & (reviews['hostId'].isnull())]), ' records')
    reviews = reviews[(reviews['hostId'] != 'null') & (~reviews['hostId'].isnull())]
    print('Updated number of host review records:', len(reviews))

    #Drop duplicates
    print("Drop duplicate records")
    reviews = reviews.drop_duplicates()
    print('Updated number of host review records:', len(reviews))

    print('Removing records with non-numeric guest ids:')
    reviews = reviews[reviews.guestId.apply(lambda x: x.isnumeric())]
    print('Updated number of host review records:', len(reviews))

    return reviews

def rename_host_review_cols(reviews):
    return reviews.rename(columns={'hostName': 'host_name', 'totalReviewsFromHosts': 'total_host_reviews', 'guestId':'recipient_id', 'hostId': 'reviewer_id'})

def restrict_by_people_pic(pic_tbl, hosts, listings, reviews, guests):

    # Reduced Hosts
    new_host_ids = set(utils.convert_to_str(pic_tbl[pic_tbl['num_of_people_in_pic'] == 1]['id'].unique()))

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(new_host_ids)]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['host_id'].isin(new_host_ids)]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    # Reduced Reviews
    new_reviews = reviews[reviews['recipient_id'].isin(new_host_ids)]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Overall Guests
    overall_guests = reviews['reviewer_id'].unique()
    new_overall_guests = new_reviews['reviewer_id'].unique()
    print("Revised number of Overall Guests: %d (decreased %.2f %%)" % (len(new_overall_guests), utils.get_decreased_percent(new_overall_guests, overall_guests)))

    # Reduced Retrieved Guests
    new_retrieved_guests = guests[guests['id'].isin(set(utils.convert_to_str(new_overall_guests)))]
    print("Revised number of Retrieved Guests: %d (decreased %.2f %%)" % (len(new_retrieved_guests), utils.get_decreased_percent(new_retrieved_guests, guests)))

    return (new_hosts, new_listings, new_reviews, new_overall_guests, new_retrieved_guests)

def restrict_by_received_guests(hosts, listings, reviews, guests):

    # Reduced Reviews
    print('Restrict to only reviews from guests whose profile we have')
    new_reviews = reviews[reviews['reviewer_id'].isin(guests['id'].unique())]
    print("Revised number of Reviews: %d (decreased %.2f %%)" % (len(new_reviews), utils.get_decreased_percent(new_reviews, reviews)))

    # Reduced Hosts
    new_hosts = hosts[hosts['id'].isin(set(new_reviews['recipient_id'].astype(str)))]
    print("Revised number of Hosts: %d (decreased %.2f %%)" % (len(new_hosts), utils.get_decreased_percent(new_hosts, hosts)))

    # Reduced Listings
    new_listings = listings[listings['id'].isin(set(new_reviews['listing_id'].astype(str)))]
    print("Revised number of Listings: %d (decreased %.2f %%)" % (len(new_listings), utils.get_decreased_percent(new_listings, listings)))

    return (new_hosts, new_listings, new_reviews)
