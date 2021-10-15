# Shell script to download and split the sentiment140 dataset from kaggle.
# Ensure that you have the kaggle api installed in your zsh env for this to
# work...
# this also requires that you have randomize-lines install on mac (use
# homebrew `brew install randomize-lines`)

#!/usr/bin/env zsh

# Create data directory
mkdir -p data/sentiment_1400k

# Download kaggle dataset to root, unzip and move to data dir. Delete zip
kaggle datasets download kazanova/sentiment140
unzip sentiment140.zip

mv training.1600000.processed.noemoticon.csv ./data/sentiment_1400k
rm sentiment140.zip

# Go to data folder, randomize lines of csv to mix classes, split into train,
# dev, test. Tidy up
cd data/sentiment_1400k

rl training.1600000.processed.noemoticon.csv -o shuffled.csv

split -l 1400000 shuffled.csv
sleep 5s
mv xaa train.csv
mv xab interim

split -l 100000 interim
sleep 5s
mv xaa dev.csv
mv xab test.csv

rm training.1600000.processed.noemoticon.csv
rm shuffled.csv
rm interim
