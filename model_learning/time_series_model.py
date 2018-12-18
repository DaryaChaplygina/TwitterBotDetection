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
                                       random_state=17)
        prof_lr, tweet_lr = self.train_model(train)
        pred, real = self.predict(test, prof_lr, tweet_lr)
        pred_class = np.argmax(pred, axis=1)
        print(precision_recall_fscore_support(real, pred_class))

    def train_model(self, train: pd.DataFrame):
        train_prof, train_tweet_all, train_labels = self.prepare_data(train)
        profile_lr = LogisticRegression(n_jobs=-1).fit(train_prof,
                                                       train_labels)
        print("done prof")

        train_tweet, tweet_labels = self.__dropna_tweet(train_tweet_all,
                                                        train_labels)
        tweet_lr = LogisticRegression(n_jobs=-1).fit(train_tweet,
                                                     tweet_labels)
        print("done tweet")
        return profile_lr, tweet_lr

    def prepare_data(self, data: pd.DataFrame):
        profile_data, tweet_data, labels = self.__split_df(data)
        labels = np.asarray([(0 if l == "genuine" else 1) for l in labels])
        profile_data = np.asarray(
            [[0, 1000000] if profile_data[i, 0] == 0 else profile_data[i, :]
             for i in range(profile_data.shape[0])])   # replace inf

        return profile_data, tweet_data, labels

    @staticmethod
    def __split_df(data: pd.DataFrame):
        profile_data = data[['following', 'followers_to_following_ratio']]
        tweet_data = data[['links_per_tweet', 'unique_links_per_tweet',
                           'usernames_per_tweet',
                           'unique_usernames_per_tweet', 'tweets_entropy']]
        labels = data[['label']]
        return profile_data.values, tweet_data.values, labels.values

    @staticmethod
    def __dropna_tweet(tweet_data: np.ndarray, labels: np.ndarray):
        tweet = []
        tweet_labels = []
        for i in range(tweet_data.shape[0]):
            if not np.isnan(tweet_data[i, 0]):
                tweet.append(tweet_data[i, :])
                tweet_labels.append(labels[i])

        return np.asarray(tweet), np.asarray(tweet_labels)

    def predict(self, test: pd.DataFrame, prof_lr, tweet_lr):
        test_prof, test_tweet_all, test_labels = self.prepare_data(test)
        prof_pred = prof_lr.predict_proba(test_prof)

        test_tweet, _ = self.__dropna_tweet(test_tweet_all, test_labels)
        tweet_pred = tweet_lr.predict_proba(test_tweet)

        prediction = np.zeros((test_prof.shape[0], 2))
        tweet_idx = 0
        for i in range(test_prof.shape[0]):
            if np.isnan(test_tweet_all[i, 0]):
                prediction[i] = prof_pred[i]
            else:
                prediction[i] = 0.5 * prof_pred[i] \
                                + 0.5 * tweet_pred[tweet_idx]
                tweet_idx += 1

        return prediction, test_labels


if __name__ == "__main__":
    # simple test
    ts = TimeSeriesModel("../Datasets/cresci-2017/timeseries_features.csv")
    ts.run_model()
