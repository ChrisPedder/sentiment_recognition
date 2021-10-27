# sentiment_recognition
A repo for trying out some sentiment recognition work.

To get the data necessary, run the scripts `get_emotion_data.sh`and
`get_sentiment_1400k_data.sh`. For these to run, you will have to have
installed the kaggle cli tool (on mac this is as simples as `brew install kaggle`).

Next, create a virtual env with python 3.8, activate it and `pip install -r reqirements.txt`

Finally, to train a regression model, do

`python3 regression.py --job train --model_type BoW`

which will train a bag of words model for sentiment detection. To evaluate, run the same, but with `--job evaluate`.

More features added to train models for sentiment and emotion recognition
using static BERT-type embeddings...

Next up, fine tuning BERT models using huggingface...
