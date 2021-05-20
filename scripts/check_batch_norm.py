#!/usr/bin/env python3

import argparse
from project import Project
from manager import Manager



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Check maximum mean, standard deviation '
                                                'of batch norm layers.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ckpt_filename', metavar='ckpt_filename[STR]',
                        type=str, help='')
    args = parser.parse_args()

    ckpt = Project.get_loaded_ckpt(args.ckpt_filename)

    device = ckpt['params']['device']
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise OSError('cuda not available...')

    net  = Manager.build_model(ckpt['params'], device=device)
    net.load_state_dict(ckpt['model_state'], strict=True)
    net.eval()

    max_quantile_68 = 0.
    max_mean, max_std = 0., 0.
    acc_mean, acc_std = 0., 0.
    total = 0.

    for weight, bias in net.parameters_batch_norm():
        acc_mean += bias.sum().item()
        acc_std += weight.sum().item()
        total += weight.size(0)

        max, argmax = (weight + bias).max(0)
        if max > max_quantile_68:
            max_quantile_68 = max
            max_std = weight[argmax].item()
            max_mean = bias[argmax].item()

    print('\nMax 68% quantile :', max_quantile_68.item())
    print('Avg [mean/std] : [{:.3f}, {:.3f}]'.format(acc_mean/total,
                                                    acc_std/total))
    print('Max [mean/std] : [{:.3f}, {:.3f}]'.format(max_mean, max_std))
