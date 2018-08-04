import os
import pandas as pd
import glob
import settings as s

def readin_dir(path="", only_ext= None, delimiter=','):
    results = None
    try:
        #get file paths
        print("Retrieving CSV files from ", path)
        out_files = glob.glob(os.path.join(path, '*'))[::-1] if only_ext == None else glob.glob(os.path.join(path, '*.' + only_ext))[::-1]

        # concat all files and format columns
        o = []
        for f in out_files:
            print('Processing ', f)
            o.append(pd.read_csv(f, header = None, delimiter=delimiter, engine='python'))
        results = pd.concat(o)
        results = results.rename(columns=results.iloc[0]).drop(results.index[0]).reset_index(drop=True)
    except Exception as e:
        print("Error:", e)
    return results

def readin_guests(path=s.RAW_GUESTS_DIR):
    guests = None
    try:
        #get file paths
        print("Retrieving CSV files from ", path)
        out_files = glob.glob(os.path.join(path, '*.csv'))[::-1]

        # concat all files and format columns
        o = []
        for f in out_files:
            print('Processing ', f)
            o.append(pd.read_csv(f, header = None, delimiter= '==', engine='python'))
        guests = pd.concat(o)
        guests = guests.rename(columns=guests.iloc[0]).drop(guests.index[0]).reset_index(drop=True)
    except Exception as e:
        print("Error:", e)
    return guests

def readin_file(path):
    print("Retrieving CSV file from ", path)
    return pd.read_csv(path)

def save_all(prefix, guests, hosts, listings, reviews, hostTrips=None, hostReviews=None, guestIDs = None):
    guests.to_csv(prefix + '_guests.csv')
    hosts.to_csv(prefix + '_hosts.csv')
    listings.to_csv(prefix + '_listings.csv')
    reviews.to_csv(prefix + '_reviews.csv')
    if (hostTrips is not None):
        hostTrips.to_csv(prefix + '_hostTrips.csv')
    if (hostReviews is not None):
        hostReviews.to_csv(prefix + '_hostReviews.csv')
    if (guestIDs is not None):
        pd.DataFrame({'id':list(guestIDs)}).to_csv(prefix + '_fullguestIDs.csv')
