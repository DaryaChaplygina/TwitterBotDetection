import json

from dataset_creating.dataset import Dataset


class DatasetsCollector:
    def __init__(self, ds_name: str):
        """
        loads csv tables from paths listed in dataset_path_list and merges
        them into one dataset
        :param ds_name: name of dataset should be keyword in dataset_path_list
        """
        f = open('dataset_path_list.json')
        data = json.load(f)
        self._ds_files = data[ds_name]

    def collect_timeseries_dataset(self, save_path: str):
        for data in self._ds_files:
            ds = Dataset(data['path'], data['label'])
            ds.add_user_features()
            ds.add_tweet_statistical_features()
            ds.add_tweet_timeseries_features()
            ds.save(save_path)


if __name__ == "__main__":
    dc = DatasetsCollector("cresci-2017")
    dc.collect_timeseries_dataset('Datasets/cresci-2017/'
                                  'timeseries_features.csv')
