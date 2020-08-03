#!/usr/bin/env python3

import argparse
import copy
from main import main_prog
from project import Project


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description=('Replot test images.'),
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir_model', metavar='model_log_dir[STR]', type=str, help='')
    parser.add_argument('--save_fig', action='store_true', help='')
    args = parser.parse_args()

    ckpt_filename = Project.get_ckpt_from_log_dir_model(args.log_dir_model)
    _, params = Project.load_ckpt_params(ckpt_filename)

    params['mode'] = 'test'
    params['ckpt_filename'] = ckpt_filename
    if args.log_dir_model[-1] == '/':
        args.log_dir_model = args.log_dir_model[:-1]
    params['log_dir'] = '/'.join(args.log_dir_model.split('/')[:-1])
    params['plot_imgs'] = True
    params['save_imgs'] = args.save_fig
    main_prog(copy.deepcopy(params))
