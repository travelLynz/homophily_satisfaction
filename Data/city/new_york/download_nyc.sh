wget http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/calendar.csv.gz
gunzip calendar.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/reviews.csv.gz
gunzip reviews.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/listings.csv.gz
gunzip listings.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/visualisations/listings.csv -O 'listing-list.csv'
wget http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/visualisations/reviews.csv -O 'reviews-list.csv'
