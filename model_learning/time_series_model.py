import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


class TimeSeriesModel:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, encoding="utf-8")

    def run_model(self):
        train, test = train_test_split(self.data, test_size=0.3,
                                       random_state=14)
        prof_lr, tweet_lr = self.train_model(train)
        pred, real = self.predict(test, prof_lr, tweet_lr)
        pred_class = np.argmax(pred, axis=1)
        return precision_recall_fscore_support(real, pred_class)

    def train_model(self, train: pd.DataFrame):
        prof_data, prof_labels, tweet_data, tweet_labels = \
            self.prepare_data(train)

        profile_lr = LogisticRegression().fit(prof_data, prof_labels.ravel())
        tweet_lr = LogisticRegression().fit(tweet_data, tweet_labels.ravel())

        return profile_lr, tweet_lr

    @staticmethod
    def prepare_data(data: pd.DataFrame):
        profile_data = data[['following', 'followers_to_following_ratio']].copy()
        profile_data.replace(np.inf, 1000000, inplace=True)
        profile_labels = data[['label']].copy()
        profile_labels = profile_labels.replace({"genuine": 0, "bot": 1})

        tweet_data_labeled = data[['links_per_tweet', 'unique_links_per_tweet',
                                   'usernames_per_tweet', 'label',
                                   'unique_usernames_per_tweet',
                                   'tweets_entropy']].copy()

        # nan appears when no users tweets available
        tweet_data_labeled.dropna(axis=0, how='any', inplace=True)
        tweet_data = tweet_data_labeled.drop(labels=['label'], axis=1)
        tweet_labels = tweet_data_labeled['label']
        tweet_labels = tweet_labels.replace({"genuine": 0, "bot": 1})

        return profile_data.values, profile_labels.values,\
            tweet_data.values, tweet_labels.values

    def predict(self, test: pd.DataFrame, prof_lr, tweet_lr):
        prof_data, prof_labels, tweet_data, tweet_labels = \
            self.prepare_data(test)

        prof_pred = prof_lr.predict_proba(prof_data)
        tweet_pred = tweet_lr.predict_proba(tweet_data)

        prediction = np.zeros((prof_data.shape[0], 2))
        tweet_idx = 0
        for i in range(prof_data.shape[0]):
            if np.isnan(test['links_per_tweet'].iloc[i]):
                prediction[i] = prof_pred[i]
            else:
                prediction[i] = 0.5 * prof_pred[i] \
                                + 0.5 * tweet_pred[tweet_idx]
                tweet_idx += 1

        return prediction, prof_labels


if __name__ == "__main__":
    # simple test
    ts = TimeSeriesModel("../Datasets/cresci-2017/timeseries_features.csv")
    print(ts.run_model())
