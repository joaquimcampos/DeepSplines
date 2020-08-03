import os
import collections
from ds_utils import json_load, json_dump


class MyJson():
    """ Class for logging in a json file the avg/median results across runs.

    Used by SearchRun class (see scripts/search_run.py).
    """
    def __init__(self, log_dir, model_name, additional_info=None):
        """
        Args:
            log_dir: directory for json logging.
            model_name: name of model whose results are to be logged.
            additional_info: additional info to log (added to default list).
        """
        # default info to log
        self.info_list = ['avg_train_df_loss', 'avg_valid_df_loss',
                        'avg_train_acc', 'valid_acc']

        if additional_info is not None:
            try:
                self.info_list += additional_info
            except TypeError:
                self.info_list += list(additional_info)

        self.log_dir = log_dir
        self.model_name = model_name

        results_json_filename = 'avg_results.json'
        self.results_json = os.path.join(self.log_dir, results_json_filename)

        if not os.path.isfile(self.results_json):
            results_dict = {} # new dictionary
        else:
            results_dict = json_load(self.results_json)

        self.dict_info_list = ['valid_acc', 'test_acc', 'threshold',
                                'sparsity', 'lipschitz_bound']

        self.avg_info_list = ['avg_train_df_loss', 'avg_valid_df_loss', # data fidelity
                                'avg_train_tv_loss', 'avg_train_lipschitz_loss',
                                'avg_train_acc', 'avg_time', 'avg_max_memory',
                                'avg_test_loss']

        if self.model_name not in results_dict:
            results_dict[self.model_name] = {}

        if self.sorting_key not in results_dict[self.model_name]:
            results_dict[self.model_name][self.sorting_key] = {'median': 0.}

        json_dump(results_dict, self.results_json)


    @property
    def sorting_key(self):
        if 'test_acc' in self.info_list:
            return 'test_acc'
        else:
            return 'valid_acc'


    def update_json(self, info, value):
        """ """
        assert info in self.info_list, f'{info} is not in json info_list...'
        if info == self.sorting_key:
            assert isinstance(value, dict) # this is required for sorting

        # save in json
        results_dict = json_load(self.results_json)

        if isinstance(value, dict):
            assert info in self.dict_info_list
            for key, val in value.items():
                if isinstance(val, int):
                    results_dict[self.model_name][info][key] = val
                else:
                    results_dict[self.model_name][info][key] = float('{:.3f}'.format(val))
        else:
            assert info in self.avg_info_list
            results_dict[self.model_name][info] = float('{:.3f}'.format(value))


        sorted_results = sorted(results_dict.items(),
                        key=lambda kv : kv[1][self.sorting_key]['median'], reverse=True)
        sorted_results_dict = collections.OrderedDict(sorted_results)

        json_dump(sorted_results_dict, self.results_json)
