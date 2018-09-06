import numpy as np
import os
from glob import glob
import shutil
from datetime import datetime
from scipy.ndimage import imread

##
# Data
##



def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

DB_NAME = 'airbnb'

CONNECT_SQL_STRING = 'mysql+mysqlconnector://root:airbnb2018@localhost/'
CONNECT_SQLALCHEMY_STRING = 'mysql+pymysql://root:airbnb2018@localhost/airbnb?charset=utf8mb4'
# root directory for all data
DATA_DIR = get_dir('../Data/')
SCRAPER_DIR = get_dir('/Users/lynraybarends/eclipse-workspace/airbnb-info-host')
RAW_DATA_DIR = os.path.join('..', DATA_DIR, 'raw')
RAW_GUESTS_DIR = os.path.join(RAW_DATA_DIR, 'guests')
RAW_HOST_HOSTS_DIR = os.path.join(RAW_DATA_DIR, 'hostshosts')
RAW_HOST_REVIEWS_DIR = os.path.join(RAW_DATA_DIR, 'hostguestreviews')
RAW_LISTINGS_DIR = os.path.join(RAW_DATA_DIR, 'listings.csv')
RAW_REVIEWS_DIR = os.path.join(RAW_DATA_DIR, 'reviews.csv')
COGNITIVE_KEY = '206cefa4e6904d1586c2e5583316c27c'
SIGHTCORP_KEY = '1a4fb409b7fa47c6b98602d114f74a1a'
HOSTS_LAT_LNG = (38.0, -97.0)
HOSTS_CCODE = 'US'
HOSTS_REGION = 'Americas'
HOSTS_SUBREGION = 'Northern America'

# coll
listing_cols = [ 'id', 'host_id', 'name', 'summary', 'space','description','neighborhood_overview','notes',
            'transit', 'access', 'interaction', 'house_rules', 'picture_url', 'street', 'neighbourhood_cleansed',
            'neighbourhood_group_cleansed', 'zipcode', 'market', 'smart_location', 'country_code', 'latitude',
            'longitude', 'is_location_exact', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
            'beds', 'bed_type', 'amenities', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
            'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30',
            'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating',
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value', 'instant_bookable', 'is_business_travel_ready',
            'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification',
            'reviews_per_month']

host_cols = [ 'host_id', 'host_name', 'host_since', 'host_location','host_about',
            'host_response_time', 'host_response_rate',
            'host_is_superhost', 'host_neighbourhood',
            'host_listings_count', 'host_total_listings_count', 'host_verifications',
            'host_has_profile_pic', 'host_identity_verified', 'calculated_host_listings_count']
guestReview_cols = ['listing_id', 'id', 'date', 'reviewer_id', 'comments']
hostReview_cols = ['recipient_id', 'reviewer_id', 'host_name', 'comments', 'total_host_reviews']

TABLES = {
    'listings' : (
        "CREATE TABLE listings (\
            idListing int(11) NOT NULL AUTO_INCREMENT,\
            id varchar(45) NOT NULL,\
            host_id varchar(45) NOT NULL,\
            name text(2000) DEFAULT NULL,\
            summary varchar(2000) DEFAULT NULL,\
            space text(2000) DEFAULT NULL,\
            description text(2000) DEFAULT NULL,\
            neighborhood_overview text(2000) DEFAULT NULL,\
            notes text(2000) DEFAULT NULL,\
            transit text(2000) DEFAULT NULL,\
            access text(2000) DEFAULT NULL,\
            interaction text(2000) DEFAULT NULL,\
            house_rules text(2000) DEFAULT NULL,\
            picture_url varchar(200) DEFAULT NULL,\
            street varchar(200) DEFAULT NULL,\
            neighbourhood_cleansed varchar(200) DEFAULT NULL,\
            neighbourhood_group_cleansed varchar(200) DEFAULT NULL,\
            zipcode varchar(15) DEFAULT NULL,\
            market varchar(200) DEFAULT NULL,\
            smart_location varchar(500) DEFAULT NULL,\
            country_code varchar(3) DEFAULT NULL,\
            latitude varchar(200) DEFAULT NULL,\
            longitude varchar(200) DEFAULT NULL,\
            is_location_exact varchar(1) DEFAULT NULL,\
            property_type varchar(200) DEFAULT NULL,\
            room_type varchar(200) DEFAULT NULL,\
            accommodates int(20) DEFAULT NULL,\
            bathrooms double DEFAULT NULL,\
            bedrooms double DEFAULT NULL,\
            beds double DEFAULT NULL,\
            bed_type varchar(200) DEFAULT NULL,\
            amenities text(2000) DEFAULT NULL,\
            price varchar(50) DEFAULT NULL,\
            weekly_price varchar(50) DEFAULT NULL,\
            monthly_price varchar(50) DEFAULT NULL,\
            security_deposit varchar(50) DEFAULT NULL,\
            cleaning_fee varchar(50) DEFAULT NULL,\
            guests_included int(20) DEFAULT NULL,\
            extra_people varchar(10) DEFAULT NULL,\
            minimum_nights int(3) DEFAULT NULL,\
            maximum_nights int(5) DEFAULT NULL,\
            availability_30 int(2) DEFAULT NULL,\
            availability_60 int(2) DEFAULT NULL,\
            availability_90 int(2) DEFAULT NULL,\
            availability_365 int(3) DEFAULT NULL,\
            number_of_reviews int(20) DEFAULT NULL,\
            review_scores_rating double DEFAULT NULL,\
            review_scores_accuracy double DEFAULT NULL,\
            review_scores_cleanliness double DEFAULT NULL,\
            review_scores_checkin double DEFAULT NULL,\
            review_scores_communication double DEFAULT NULL,\
            review_scores_location double DEFAULT NULL,\
            review_scores_value double DEFAULT NULL,\
            instant_bookable varchar(1) DEFAULT NULL,\
            is_business_travel_ready varchar(1) DEFAULT NULL,\
            cancellation_policy varchar(200) DEFAULT NULL,\
            require_guest_profile_picture varchar(1) DEFAULT NULL,\
            require_guest_phone_verification varchar(1) DEFAULT NULL,\
            reviews_per_month double DEFAULT NULL,\
            PRIMARY KEY (idListing),\
            KEY secondary (id)\
        ) ENGINE=InnoDB AUTO_INCREMENT=0"),
    'hosts': (
        "CREATE TABLE hosts (\
            id int(11) NOT NULL,\
            name varchar(200) DEFAULT NULL,\
            since varchar(15) DEFAULT NULL,\
            location varchar(500) DEFAULT NULL,\
            about text(20000) DEFAULT NULL,\
            response_time varchar(50) DEFAULT NULL,\
            response_rate varchar(4) DEFAULT NULL,\
            is_superhost varchar(1) DEFAULT NULL,\
            neighbourhood varchar(2000) DEFAULT NULL,\
            listings_count double DEFAULT NULL,\
            total_listings_count double DEFAULT NULL,\
            verifications varchar(2000) DEFAULT NULL,\
            has_profile_pic varchar(1) DEFAULT NULL,\
            identity_verified varchar(1),\
            calculated_listings_count int(20) DEFAULT NULL\
        ) ENGINE=InnoDB;"),
    'guests': (
        "CREATE TABLE guests (\
            idGuest int(11) NOT NULL AUTO_INCREMENT,\
            id varchar(200) NOT NULL ,\
            name varchar(200) DEFAULT NULL ,\
            city varchar(200) DEFAULT NULL ,\
            ccode varchar(3) DEFAULT NULL ,\
            membershipMonth varchar(20) DEFAULT NULL ,\
            membershipYear varchar(4) DEFAULT NULL ,\
            superhost varchar(5) DEFAULT NULL ,\
            verified varchar(5) DEFAULT NULL ,\
            description text(20000) DEFAULT NULL ,\
            linkedAccountVerified varchar(2000) DEFAULT NULL ,\
            schoolInfo varchar(2000) DEFAULT NULL ,\
            jobInfo varchar(2000) DEFAULT NULL ,\
            languages varchar(2000) DEFAULT NULL ,\
            reviewNumber int(25) DEFAULT NULL ,\
            guideNumber int(25) DEFAULT NULL ,\
            wishListNumber int(200) DEFAULT NULL,\
            PRIMARY KEY (idGuest)\
        ) ENGINE=InnoDB AUTO_INCREMENT=0"),
    'hostTrips': (
        "CREATE TABLE hostTrips (\
            idHostTrip int(11) NOT NULL AUTO_INCREMENT,\
            hostId varchar(200) NOT NULL ,\
            visited varchar(500) DEFAULT NULL ,\
            min_times int(100) DEFAULT NULL ,\
            city varchar(200) DEFAULT NULL ,\
            ccode varchar(3) DEFAULT NULL ,\
            country varchar(200) DEFAULT NULL ,\
            state varchar(50) DEFAULT NULL ,\
            PRIMARY KEY (idHostTrip)\
        ) ENGINE=InnoDB AUTO_INCREMENT=0"),
    'guestReviews': (
        "CREATE TABLE guestReviews (\
            idGuestReview int(11) NOT NULL AUTO_INCREMENT,\
            id varchar(45) NOT NULL ,\
            date varchar(200) DEFAULT NULL,\
            reviewer_id varchar(45) DEFAULT NULL,\
            listing_id varchar(45) DEFAULT NULL,\
            recipient_id varchar(45) DEFAULT NULL,\
            comments text(10000) DEFAULT NULL,\
            hostCancelled varchar(1) DEFAULT NULL,\
            PRIMARY KEY (idGuestReview),\
            KEY secondary (reviewer_id),\
            KEY third (id)\
        ) ENGINE=InnoDB AUTO_INCREMENT=0"),
    'hostReviews': (
        "CREATE TABLE hostReviews (\
            idHostReview int(11) NOT NULL AUTO_INCREMENT,\
            host_name varchar(200) DEFAULT NULL ,\
            reviewer_id varchar(45) DEFAULT NULL,\
            recipient_id varchar(45) DEFAULT NULL,\
            comments text(10000) DEFAULT NULL,\
            total_host_reviews text(10000) DEFAULT NULL,\
            PRIMARY KEY (idHostReview),\
            KEY secondary (reviewer_id)\
        ) ENGINE=InnoDB AUTO_INCREMENT=0")
    }

months_translated = {
    'agosto': 'August',
    'aprile': 'April',
    'dicembre': 'December',
    'febbraio': 'February',
    'gennaio': 'January',
    'giugno': 'June',
    'luglio': 'July',
    'maggio': 'May',
    'marzo': 'March',
    'novembre': 'November',
    'ottobre': 'October',
    'settembre': 'September'
}

states = {
        'Alaska',
        'Alabama',
        'Arkansas',
        'American Samoa',
        'Arizona',
        'California',
        'Colorado',
        'Connecticut',
        'District of Columbia',
        'Delaware',
        'Florida',
        'Georgia',
        'Guam',
        'Hawaii',
        'Iowa',
        'Idaho',
        'Illinois',
        'Indiana',
        'Kansas',
        'Kentucky',
        'Louisiana',
        'Massachusetts',
        'Maryland',
        'Maine',
        'Michigan',
        'Minnesota',
        'Missouri',
        'Northern Mariana Islands',
        'Mississippi',
        'Montana',
        'National',
        'North Carolina',
        'North Dakota',
        'Nebraska',
        'New Hampshire',
        'New Jersey',
        'New Mexico',
        'Nevada',
        'New York',
        'Ohio',
        'Oklahoma',
        'Oregon',
        'Pennsylvania',
        'Puerto Rico',
        'Rhode Island',
        'South Carolina',
        'South Dakota',
        'Tennessee',
        'Texas',
        'Utah',
        'Virginia',
        'Virgin Islands',
        'Vermont',
        'Washington',
        'Wisconsin',
        'West Virginia',
        'Wyoming'
}
