import re
from datetime import datetime
import pandas as pd
import numpy as np


class Tweets:
    def __init__(self, path: str):
        self._data_frame = pd.read_csv(path, encoding="utf-8")

    def get_user_tweets(self, uid: int):
        return self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['text'].values

    def get_user_link_per_tweet(self, uid: int):
        links = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['num_urls']

        return links.mean()

    def get_user_unames_per_tweet(self, uid: int):
        unames = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['num_mentions']

        return unames.mean()

    def get_user_unique_link_ratio(self, uid: int):
        link_regexp = 'https?:[^\s]+'
        tweets = self.get_user_tweets(uid)
        if len(tweets) == 0:
            return 0
        all_links = []
        for tweet in tweets:
            if not isinstance(tweet, str):
                continue
            all_links += re.findall(link_regexp, tweet)

        return len(set(all_links)) / len(tweets)

    def get_user_unique_unames_ratio(self, uid: int):
        uname_regexp = '@[^\s]+'
        tweets = self.get_user_tweets(uid)
        if len(tweets) == 0:
            return 0
        all_unames = []
        for tweet in tweets:
            if not isinstance(tweet, str):
                continue
            all_unames += re.findall(uname_regexp, tweet)

        return len(set(all_unames)) / len(tweets)

    def get_ti_entropy(self, uid: int):
        bin_len = 43200
        ts = self.get_timestamps(uid)
        if len(ts) == 0:
            return 0
        intervals = []
        for i in range(1, len(ts)):
            diff = ts[i] - ts[i - 1]
            if diff.days != -1:
                intervals.append(diff.days*24 + int(diff.seconds/(60*60)))
            else:
                intervals.append(int(diff.seconds/(60*60)))
        intervals = np.asarray(intervals)
        
        entropy = 0
        min_, max_ = np.min(intervals), np.max(intervals)
        for i in range(min_, max_ + 1):
            prob_i = np.count_nonzero(intervals == i) / len(intervals)
            if prob_i == 0:
                continue
            entropy += prob_i * np.log(prob_i)
            
        return -entropy
        

    def get_timestamps(self, uid: int):
        timestamps = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['timestamp'].values
        timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in
                      timestamps]
        return timestamps


if __name__ == "__main__":
    # simple test
    t = Tweets('/home/dario/Diploma/Datasets/cresci-2017/datasets_full.csv'
               '/social_spambots_3.csv/tweets.csv')
    t.get_ti_entropy(16282004)