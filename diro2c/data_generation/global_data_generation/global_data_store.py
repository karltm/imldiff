
class GlobalDataStore(object):
    _instance = None
    _dict_natural_datasets_diff = None
    _dict_genetic_random_datasets_diff = None

    def __init__(self):
        raise RuntimeError('Call instance() instead.')

    @classmethod
    def instance(cls):
        # singleton pattern: enable one time dataset_diff generation and provide same dataset for evaluation
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Put any initialization here.

        return cls._instance

    def load_dict_natural_dataset_diff(self, key):
        if not self._dict_natural_datasets_diff is None:
            if key in self._dict_natural_datasets_diff:
                return self._dict_natural_datasets_diff[key]

        return None

    def store_dict_natural_dataset_diff(self, key, dict_natural_dataset_diff):
        if self._dict_natural_datasets_diff is None:
            self._dict_natural_datasets_diff = {key: dict_natural_dataset_diff}
        else:
            self._dict_natural_datasets_diff[key] = dict_natural_dataset_diff

        return self._dict_natural_datasets_diff[key]

    def load_dict_genetic_random_dataset_diff(self, key):
        if not self._dict_genetic_random_datasets_diff is None:
            if key in self._dict_genetic_random_datasets_diff:
                return self._dict_genetic_random_datasets_diff[key]

        return None

    def store_dict_genetic_random_dataset_diff(self, key, dict_genetic_random_dataset_diff):
        if self._dict_genetic_random_datasets_diff is None:
            self._dict_genetic_random_datasets_diff = {
                key: dict_genetic_random_dataset_diff}
        else:
            self._dict_genetic_random_datasets_diff[key] = dict_genetic_random_dataset_diff

        return self._dict_genetic_random_datasets_diff[key]
