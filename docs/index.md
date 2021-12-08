# Sentiment Analysis using Neural Networks and BERT

## TEAM MEMBERS
*Nick Taylor
*Tatsuhi Matsumoto
*Kelly Wang
*Gwyneth Douglas
*Matthew Withey
*Nicky Dahl

## INTRODUCTION

## OVERVIEW OF DATASET
The dataset we chose for this project is a set of 1.6 million tweets that were categorized as negative or positive based on the emojis used in the tweet. The dataset is found in the Kaggle website under the title, ‘Sentiment140 Dataset with 1.6 Million Tweets.’ The target column uses the number 4 to categorize positive tweets and number 0 to categorize negative tweets. There are five other features in the dataset. The ‘id’ column contains a list of ids for the tweets. The ‘date’ column has the date, time, and timezone the tweet was tweeted. There is a ‘flag’ column that only contains the input “NO_QUERY” in every row. The next two features are the ‘user’ column containing the username of the person who tweeted the tweet and the ‘text’ column which contains the tweet itself.

![dataset](https://user-images.githubusercontent.com/45761912/145094639-ab6e083b-766c-433a-ad2c-65bc6f56fc3c.png)

## DATA PREPROCESSING
The first things we did were drop the ‘flag’ column, because it contains only one unique value, and change the positive sentiment value in the ‘target’ column from 4 to 1 to be more intuitive. To clean up the text data, we defined a list of stopwords and removed them from the tweets. Stopwords are any commonly used words that are filtered out in natural language preprocessing steps. Consisting of common words such as  “the”, “a”, “an”, and “so”, stopwords don't provide much context or information about text and are therefore removed. Next, we stripped the tweets of URLs, @ signs, non-ascii characters, & signs, greater-than, and less-than signs. Then, we set the tweets to all lower-case and separated punctuation marks from the words themselves. We also replaced many common abbreviations to the words they represent such as ‘u’ and ‘you’ and ‘r’ and ‘are’. We then separated the values in the ‘date’ column into separate features for the month, day, time, and weekday hoping that these features would be more useful. The year column was not included because all of the tweets are from the year 2009. The next pre-processing method we used was stemming, a process where words are reduced to their derived stems in order to address grammatical inflection. We also used lemmatization to simplify words by removing their inflectional endings and converting them into their base representations (lemma). 

In an effort to analyze the data, we tried some data visualization techniques. Shown below are word clouds created from the dataset. The top word cloud was created from the negative tweets and the bottom one from the positive tweets. While there are a few common words that are shared by both clouds such as “day,” “now,” “going,” “time,” and “today,” the more interesting words are the ones that can obviously hold a certain sentiment behind them. In the negative word cloud some of these that stick out are “work,” “sad,” “bad,” and “feel.” In the positive word cloud there are obvious ones like “thank,” “good,” “love,” and “great.” We were initially not sure about the accuracy of a dataset that based sentiment solely on emojis, but these word clouds represent the accuracy of the dataset by showing the pattern between the target values and the types of words used in the tweets.

![negative](https://user-images.githubusercontent.com/45761912/145095292-2e9e04a0-3e3d-4fd9-a53d-26cf0343613b.png)

![positive](https://user-images.githubusercontent.com/45761912/145095286-ead6f47d-8070-40de-b985-f1784ab8613d.png)

To gain a better understanding of the distribution of tweets in the dataset, we made a pie chart showing the percentage of tweets on each day of the week. From the pie chart, we observed that there are more tweets from the weekend in our data set. There are also more tweets from Monday than any other weekday. 

![tweets_in_a_week](https://user-images.githubusercontent.com/45761912/145095447-32c47488-1a83-4542-a802-1224a25fc3a8.jpg)

## MODELS
### Neural Network
One model that we used for our dataset was a neural network using tensorflow. We first used the Keras to tokenize the tweet strings. This isolated the 500 words that were most important to discern positive vs. negative sentiment in the dataset. After tokenizing the tweets, we defined a Keras Embedding layer to create a dense vector representation of the words along with their relative meanings. Word embeddings are an improvement over the traditional bag-of-words encoding models, as these often produce sparse vector representations consisting of mostly zero values. In word embedding, the dense vector representations are the projections of the words into a continuous vector space. The embedding or position of the word in the vector space is learned from the text and is based on the words that surround it. Using the Keras Embedding layer, we were able to test different values for the size of the vector space where the words were embedded. The next layer we defined was long short term memory (LSTM), a type of recurrent neural network for processing sequences of text data. These networks learn from memorized patterns and dependencies in the input text data. This information is then used to make predictions on or classify the sequences of data. We then used a Dense layer, which is a deeply connected layer, followed by an Activation layer. This activation layer applied a ReLU (rectified linear unit) activation function. This helps mitigate the exponential growth in computations of our neural network. We then defined a dropout layer to prevent the model from overfitting. This was followed by another set of Dense and Activation layers, except for this set, the activation layer applied a sigmoid function. This is because we wanted our output to be given as a number between 0 and 1, so that we can classify the output as positive or negative sentiment. We did so using a threshold of 0.5, where outputs less than 0.5 indicated negative sentiment and outputs greater than or equal to 0.5 indicated positive sentiment.
Our model predicted with an accuracy of 0.7969, with 81% of the tweets with negative sentiment being correctly predicted, and 78% of the tweets with positive sentiment being correctly predicted; this can be seen in our confusion matrix.

![confusion_matrix](https://user-images.githubusercontent.com/45761912/145101243-6987409d-ee2b-4c85-85d4-820b10bcdc67.png)

Our ROC curve depicted below also reflects this accuracy, as our AUC is .80, or .7969 rounded to the nearest tenth. 
![roc_curve](https://user-images.githubusercontent.com/45761912/145101255-7a2d4989-3966-4447-afb5-e39ce2944b7d.png)

One observation we made about our dataset after using the neural network model is that there seems to be a correlation between the day of the week and the ratio of positive to negative tweets. We hypothesized that Monday would have the highest ratio of negative to positive sentiment in the dataset, but found that it was actually Wednesday and Thursday that had the highest ratio of negative to positive sentiment.

![train_ratio](https://user-images.githubusercontent.com/45761912/145103707-38beb06e-1bbc-4394-ae24-8e337d31a9bd.png)
![test_ratio](https://user-images.githubusercontent.com/45761912/145103712-e71a3407-d35f-4ef0-b7eb-1bf92e0cc833.png)

Another observation that we made was that increasing the amount of epochs from 6 to 30 had little effect on the accuracy of the model, but decreasing the amount of epochs from 6 to 1 worsened the model accuracy.

### BERT
We also used Google’s BERT (Bidirectional Encoder Representations from Transformers) model to conduct sentiment analysis on the dataset of tweets we used. BERT uses Transformer, which is a mechanism that learns contextual relations between words in text. This model is considered bidirectional because it reads an entire sequence of words at once, as opposed to a directional model that reads a sequence of words from left to right or vice versa. This attribute of BERT allows it to make full use of a word's surroundings to learn the context for which it is used. 

![bertfigure](https://user-images.githubusercontent.com/45768742/145123628-cbbe9c55-1d25-4f5e-8a4a-6bd388a1ba4d.png)

BERT makes use of two training strategies: Masked LM (MLM) and Next Sentence Prediction (NSP), where both are trained together with the goal of reducing the combined loss function of the two strategies. MLM is a strategy that involves replacing 15% of the words in every sequence with a MASK token. Based on the context of the non-masked words in the sequence, MLM attempts to predict what the masked words originally were. These predictions of the masked words are then fed into BERT’s loss function. For NSP, pairs of sentences are fed into the model, and the trainer predicts if the second sentence in the pair follows the first one in the original text or not. Half the pairs have the second sentence be the true second sentence and the other half of the pairs have a second sentence taken randomly from other sentences. The combined loss functions of these two strategies are then minimized.

Using our dataset with BERT required us to do some additional preprocessing to ensure that it was formatted correctly for the model. Since BERT is a text prediction model, we had to drop all additional features other than the tweet text itself and also the target values. This newly formatted Pandas dataframe had to then be split into training set, validation set and test set, with a 70/30 split on training/validation and test and then a 80/20 split on the test and validation sets. The BERT model utilizes TensorFlow datasets, so we had to convert our three dataframes into TensorFlow datasets that were usable by the model. We fit the model with 1 epoch and a learning rate of .00003. We ended up with a validation accuracy of 0.8661 and a test accuracy of 0.8645.

Overall the results were pretty good, though we could have had more gains in accuracy with an increase in epochs. We used the base BERT model, which is a 12-layer, 110 million parameter neural network and had fairly good results, but there is also a large BERT model that is a 24-layer, 340 million parameter neural network that is likely to have yielded even better results.

## CONCLUSION

## REFERENCES
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

https://research.google/pubs/pub47751/
