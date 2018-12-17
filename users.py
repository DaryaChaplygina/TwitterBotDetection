import pandas as pd


class Users:
    def __init__(self, path_df: str):
        self._data_frame = pd.read_csv(path_df, encoding="utf-8")
        self._current_path = path_df

    def get_uids(self):
        return self._data_frame['id']

    def get_names(self):
        return self._data_frame['name']

    def get_screen_names(self):
        return self._data_frame['screen_name']

    def get_following(self):
        return self._data_frame['friends_count']

    def followers_to_following_ratio(self):
        return self._data_frame['followers_count']\
            .divide(self._data_frame['friends_count'])


if __name__ == "__main__":
    # simple test
    u = Users('/home/dario/Diploma/Datasets/cresci-2017/datasets_full.csv'
              '/genuine_accounts.csv/users.csv')
    print(u.get_names())
