# Sentiment Analysis using Neural Network and BERT
## INTRODUCTION
## OVERVIEW OF DATASET
The dataset we chose for this project is a set of 1.6 million tweets that were categorized as negative or positive based on the emojis used in the tweet. The dataset is found in the Kaggle website under the title, ‘Sentiment140 Dataset with 1.6 Million Tweets.’ The target column uses the number 4 to categorize positive tweets and number 0 to categorize negative tweets. There are five other features in the dataset. The ‘id’ column contains a list of ids for the tweets. The ‘date’ column has the date, time, and timezone the tweet was tweeted. There is a ‘flag’ column that only contains the input “NO_QUERY” in every row. The next two features are the ‘user’ column containing the username of the person who tweeted the tweet and the ‘text’ column which contains the tweet itself.

![dataset](https://user-images.githubusercontent.com/45761912/145094639-ab6e083b-766c-433a-ad2c-65bc6f56fc3c.png)

## DATA PREPROCESSING
The first things we did were drop the ‘flag’ column, because it contains only one unique value, and change the positive sentiment value in the ‘target’ column from 4 to 1 to be more intuitive. To clean up the text data, we defined a list of stopwords and removed them from the tweets. Stopwords are any commonly used words that are filtered out in natural language preprocessing steps. Consisting of common words such as  “the”, “a”, “an”, and “so”, stopwords don't provide much context or information about text and are therefore removed. Next, we stripped the tweets of URLs, @ signs, non-ascii characters, & signs, greater-than, and less-than signs. Then, we set the tweets to all lower-case and separated punctuation marks from the words themselves. We also replaced many common abbreviations to the words they represent such as ‘u’ and ‘you’ and ‘r’ and ‘are’. We then separated the values in the ‘date’ column into separate features for the month, day, time, and weekday hoping that these features would be more useful. The year column was not included because all of the tweets are from the year 2009. The next pre-processing method we used was stemming, a process where words are reduced to their derived stems in order to address grammatical inflection. 

In an effort to analyze the data, we tried some data visualization techniques. Shown below are word clouds created from the dataset. The top word cloud was created from the negative tweets and the bottom one from the positive tweets. While there are a few common words that are shared by both clouds such as “day,” “now,” “going,” “time,” and “today,” the more interesting words are the ones that can obviously hold a certain sentiment behind them. In the negative word cloud some of these that stick out are “work,” “sad,” “bad,” and “feel.” In the positive word cloud there are obvious ones like “thank,” “good,” “love,” and “great.” We were initially not sure about the accuracy of a dataset that based sentiment solely on emojis, but these word clouds represent the accuracy of the dataset by showing the pattern between the target values and the types of words used in the tweets.

![negative](https://user-images.githubusercontent.com/45761912/145095292-2e9e04a0-3e3d-4fd9-a53d-26cf0343613b.png)

![positive](https://user-images.githubusercontent.com/45761912/145095286-ead6f47d-8070-40de-b985-f1784ab8613d.png)

To gain a better understanding of the distribution of tweets in the dataset, we made a pie chart showing the percentage of tweets on each day of the week. From the pie chart, we observed that there are more tweets from the weekend in our data set. There are also more tweets from Monday than any other weekday. 

![tweets_in_a_week](https://user-images.githubusercontent.com/45761912/145095447-32c47488-1a83-4542-a802-1224a25fc3a8.jpg)

## MODELS
### Neural Network
### BERT
## CONCLUSION
