import argparse
import os
import json
from ds_utils import ArgCheck, json_load, json_dump


class SearchRun():
    """ Helper class for abstracting common operations to
    log runs/grid searches of models with deepspline and
    standard activations.
    """

    def __init__(self, args):
        """ """
        self.args = args



    @staticmethod
    def add_default_args(parser):
        """ Add default args, common to deepspline and
        standard model runs/grid searches.
        """
        parser.add_argument('log_dir', metavar='log_dir[STR]', type=str,
                            help='Log directory for models.')
        parser.add_argument('--start_idx', metavar='INT,>=0', type=ArgCheck.nn_int,
                            help='Start from this search/run idx.')
        parser.add_argument('--end_idx', metavar='INT,>0', type=ArgCheck.p_int,
                            help='End in this search/run idx.')
        parser.add_argument('--resume', action='store_true',
                            help='Resume search/run from last search idx.')

        return parser



    def default_params(self):
        """ Return default params, common to deepspline and
        standard model runs/grid searches.
        """
        params = {'log_dir': self.args.log_dir,
                'resume': self.args.resume}

        return params



    def init_indexes(self, log_dir, num_iter):
        """ init json for logging run/search index
        and get index range for runs/search loop.

        Args:
            log_dir: directory for logging index.
            num_iter: number of runs/search iterations.
        Returns:
            (start_idx, end_idx): run/search range indexes.
        """
        self.init_index_json(log_dir)
        start_idx, end_idx = self.get_index_range(num_iter)

        return start_idx, end_idx



    def init_index_json(self, log_dir):
        """ Init json for logging run/search index,
        allowing to resume training later.
        """
        self.index_json = os.path.join(log_dir, 'index.json')



    def update_index_json(self, idx):
        """ Update current run/search idx.
        """
        json_dump({'last_idx' : idx, 'end_idx' : self.end_idx},
                self.index_json)



    def get_index_range(self, num_iter):
        """ Get index range for runs/search loop.

        If resuming, get the last run/search index from
        index json file and continue from there;
        else, if provided by user, use start_idx and/or end_idx;
        otherwise, set start_idx = 0, end_idx = num_iter.
        """
        start_idx = 0
        end_idx = num_iter

        if self.args.resume is True:
            search_dict = json_load(self.index_json)
            start_idx = search_dict['last_idx']
            end_idx = search_dict['end_idx']
            print(f'==> Run idx {start_idx}-{end_idx}')

        elif self.args.start_idx is not None:
            if self.args.start_idx < num_iter:
                start_idx = self.args.start_idx
            if self.args.end_idx is not None and self.args.end_idx < num_iter:
                end_idx = self.args.end_idx

            print(f'==> Run idx {start_idx}-{end_idx}')

        self.end_idx = end_idx

        return start_idx, end_idx
