from dataset import Dataset
import json


class DatasetsCollector:
    def __init__(self, ds_name):
        """
        loads csv tables from paths listed in dataset_path_list and merges
        them into one dataset
        :param ds_name: name of dataset should be keyword in dataset_path_list
        """
        f = open('dataset_path_list.json')
        data = json.load(f)
        self._ds_files = data[ds_name]

    def collect_all(self, save_path):
        for [path, label] in self._ds_files:
            ds = Dataset(path, label)
            ds.add_user_features()
            ds.add_tweet_statistical_features()
            ds.save(save_path)


if __name__ == "__main__":
    fc = DatasetsCollector("cresci-2017")
    fc.collect_all('Datasets/cresci-2017/all_features.csv')
