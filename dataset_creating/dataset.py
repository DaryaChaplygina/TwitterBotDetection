import os

import pandas as pd

from text_processor_simple import TextProcessor
from tweets import Tweets
from users import Users


class Dataset:
    def __init__(self, path: str, label=""):
        self._path = path
        self._dataset = pd.DataFrame()

        users = Users(self._path + "users.csv")
        self._dataset['id'] = users.get_uids()
        self._dataset = self._dataset.assign(label=label)

    def add_user_features(self):
        users = Users(self._path + "users.csv")

        self._dataset['name'] = users.get_names()
        self._dataset['screen_name'] = users.get_screen_names()
        self._dataset['following'] = users.get_following()
        self._dataset['followers_to_following_ratio'] = \
            users.followers_to_following_ratio()

    def add_tweet_statistical_features(self):
        tweets = Tweets(self._path + "tweets.csv")

        self._dataset['links_per_tweet'] = self._dataset['id']\
            .apply(tweets.get_user_link_per_tweet)
        self._dataset['unique_links_per_tweet'] = self._dataset['id']\
            .apply(tweets.get_user_unique_link_ratio)
        self._dataset['usernames_per_tweet'] = self._dataset['id']\
            .apply(tweets.get_user_unames_per_tweet)
        self._dataset['unique_usernames_per_tweet'] = self._dataset['id']\
            .apply(tweets.get_user_unique_unames_ratio)

    def add_tweet_text_features(self):
        tweets = Tweets(self._path + "tweets.csv")
        tp = TextProcessor()
        self._dataset['tweets'] = self._dataset['id']\
            .apply(lambda id: [tp.preprocess(tweet) for tweet in
                               tweets.get_user_tweets(id)])
        
    def add_tweet_timeseries_features(self):
        tweets = Tweets(self._path + "tweets.csv")
        self._dataset['tweets_entropy'] = self._dataset['id']\
            .apply(tweets.get_ti_entropy)

    def save(self, save_path: str):
        if os.path.isfile(save_path):
            self._dataset.to_csv(save_path, mode='a', header=False)
        else:
            self._dataset.to_csv(save_path)

    def merge_with(self, path1: str, name: str):
        df = pd.read_csv(path1)
        res = pd.merge(self._dataset, df, on='id', how='left')
        res.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        res.to_csv(self._path + name + ".csv")


if __name__ == "__main__":
    path = '/home/dario/Diploma/Datasets/cresci-2017/datasets_full.csv' \
           '/social_spambots_3.csv/'
    ds = Dataset(path)
    ds.add_tweet_text_features()
    # print(ds.head(1))
    # ds.add_user_features()
    # ds.save('test_ds')