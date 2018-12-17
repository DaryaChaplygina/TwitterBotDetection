import pandas as pd
import re


class Tweets:
    def __init__(self, path):
        self._data_frame = pd.read_csv(path, encoding="utf-8")

    def get_user_tweets(self, uid):
        return self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['text'].values

    def get_user_link_per_tweet(self, uid):
        links = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['num_urls']

        return links.mean()

    def get_user_unames_per_tweet(self, uid):
        unames = self._data_frame\
            .loc[self._data_frame['user_id'] == uid]['num_mentions']

        return unames.mean()

    def get_user_unique_link_ratio(self, uid):
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

    def get_user_unique_unames_ratio(self, uid):
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


if __name__ == "__main__":
    # simple test
    t = Tweets('/home/dario/Diploma/Datasets/cresci-2017/datasets_full.csv'
               '/genuine_accounts.csv/tweets.csv')
    print(t.get_user_tweets(2257926470))