rm *.csv
wget http://data.insideairbnb.com/united-states/ny/new-york-city/$1/data/calendar.csv.gz
gunzip calendar.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/$1/data/reviews.csv.gz
gunzip reviews.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/$1/data/listings.csv.gz
gunzip listings.csv.gz
wget http://data.insideairbnb.com/united-states/ny/new-york-city/$1/visualisations/listings.csv -O 'listing-list.csv'
wget http://data.insideairbnb.com/united-states/ny/new-york-city/$1/visualisations/reviews.csv -O 'reviews-list.csv'
