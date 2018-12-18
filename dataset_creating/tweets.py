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
        
        entropy = count_entropy(intervals)
        print(uid, "+")                     # progress output
        return entropy
        

    def get_timestamps(self, uid: int):
        timestamps = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['timestamp'].values
        timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in
                      timestamps]
        return timestamps


def count_entropy(series: np.ndarray):
    r = 1
    N = series.shape[0]

    def _dist_matrix(xs, m):
        dist_matrix = np.zeros((N - m + 1, N - m + 1))
        for i in range(N - m + 1):
            for j in range(i, N - m + 1):
                dist_matrix[i, j] = dist_matrix[j, i] = \
                    max([abs(xs[i][k] - xs[j][k]) for k in range(m)])
        return dist_matrix

    def _phi(m):
        xs = [series[i:i+m] for i in range(N - m + 1)]
        dist_matrix = _dist_matrix(xs, m)

        C_m = []
        for i in range(N - m + 1):
            n_less = len(list(filter(lambda x: x <= r, dist_matrix[i, :])))
            if n_less == 0:
                continue
            C_m.append(np.log(n_less / (N - m + 1)))

        return sum(C_m) / (N - m + 1)

    return _phi(2) - _phi(3)


if __name__ == "__main__":
    # simple test
    t = Tweets('/home/dario/Diploma/Datasets/cresci-2017/datasets_full.csv'
               '/social_spambots_3.csv/tweets.csv')
    print(t.get_ti_entropy(16282004))