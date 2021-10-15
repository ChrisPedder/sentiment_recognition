# Shell script to download emotions-dataset-for-nlp dataset from kaggle.
# Ensure that you have the kaggle api installed in your zsh env for this to
# work...

#!/usr/bin/env zsh

# Create data directory
mkdir -p data/emotion

# Download kaggle dataset to root, unzip and move to data dir. Delete zip
kaggle datasets download praveengovi/emotions-dataset-for-nlp
unzip emotions-dataset-for-nlp.zip

mv train.txt ./data/emotion
mv val.txt ./data/emotion
mv test.txt ./data/emotion

rm emotions-dataset-for-nlp.zip
