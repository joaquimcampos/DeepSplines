import os
import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
from ds_utils import check_device, denormalize


def init_dataset(**params):

    # add your datasets here and create a corresponding Dataset class
    dataset_dict = {'s_shape_1500' : S_shape, 'circle_1500' : Circle,
                    'cifar10' : Cifar10, 'cifar100' : Cifar100, 'mnist' : MNIST}

    if params['dataset_name'] not in dataset_dict.keys():
        raise ValueError('Chosen dataset is not available... Exiting.')

    dataset = dataset_dict[params['dataset_name']](**params)

    return dataset



class Dataset():

    def __init__(self, dataset_name=None, log_dir=None, model_name=None,
                plot_imgs=False, save_imgs=False, save_title=None, **kwargs):

        self.name = dataset_name

        if model_name is not None:
            self.log_dir_model = os.path.join(log_dir, model_name)
        else:
            self.log_dir_model = log_dir

        self.plot_imgs = plot_imgs
        self.save_imgs = save_imgs
        self.get_plot = plot_imgs or save_imgs
        self.save_title = save_title


    def plot_train_imgs(self, *args, **kwargs):
        """ """
        raise NotImplementedError



class TorchDataset(Dataset):

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.is_user_dataset = False
        self.sample_batch = 4 # show 4 sample images


    def get_minimal_transform_list(self):
        """ Get minimal transform list (applied to train, validation and test sets)
        """
        normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        minimal_transform_list = [transforms.ToTensor(), normalize]

        return minimal_transform_list


    def get_augment_transform_list(self):
        """ Gets list of training augmentation transforms
        """
        raise NotImplementedError


    def get_train_valid_transforms(self):
        """ Get training and validation data transforms
        """
        # augment training data
        train_transform_list = self.get_augment_transform_list()
        minimal_transform_list = self.get_minimal_transform_list()

        train_transform = transforms.Compose(train_transform_list + minimal_transform_list)
        valid_transform = transforms.Compose(minimal_transform_list)

        return train_transform, valid_transform


    def get_test_transform(self):
        """ """
        minimal_transform_list = self.get_minimal_transform_list()
        test_transform = transforms.Compose(minimal_transform_list)

        return test_transform


    def get_torchvision_dataset(self):
        """ """
        raise NotImplementedError


    def plot_train_imgs(self, trainloader):
        """ """
        data_iter = iter(trainloader)
        images, labels = data_iter.next()
        images = denormalize(images, self.norm_mean, self.norm_std).clamp(min=0, max=1)
        self.plot_torch_dataset_samples(images[0:self.sample_batch], labels[0:self.sample_batch])


    def plot_torch_dataset_samples(self, images, true_labels, pred_labels=None):
        """ Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
        """
        assert math.sqrt(images.size(0)) % 1 == 0, 'plot_images expects a batch size which is a perfect power'
        H = int(math.sqrt(images.size(0)))
        images = images.numpy().transpose([0, 2, 3, 1]) # convert to numpy format (N x H x W x C)
        fig, axes = plt.subplots(H, H)

        for i, ax in enumerate(axes.flat):
            # plot img
            ax.imshow(images[i, :, :, :], interpolation='spline16')

            # show true & predicted labels
            true_labels_name = self.classes[true_labels[i]]
            if pred_labels is None:
                xlabel = "{0} ({1})".format(true_labels_name, true_labels[i])
            else:
                pred_labels_name = self.classes[pred_labels[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    true_labels_name, pred_labels_name
                )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()



class Cifar10(TorchDataset):

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.num_classes = 10
        self.norm_mean = (0.4914, 0.4822, 0.4465)
        self.norm_std  = (0.2470, 0.2435, 0.2616)
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')


    def get_augment_transform_list(self):
        """ Gets list of training augmentation transforms
        """
        train_transform_list = [transforms.RandomCrop(32, padding=4)]
        train_transform_list += [transforms.RandomHorizontalFlip()]

        return train_transform_list


    def get_torchvision_dataset(self):
        """ """
        return torchvision.datasets.CIFAR10



class Cifar100(TorchDataset):

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.num_classes = 100
        self.norm_mean = (0.5072, 0.4867, 0.4412)
        self.norm_std  = (0.2673, 0.2564, 0.2762)
        self.classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
                        'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly',
                        'camel', 'can', 'castle', 'caterpillar', 'cattle',
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                        'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                        'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                        'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                        'plain', 'plate', 'poppy', 'porcupine', 'possum',
                        'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                        'rose', 'sea', 'seal', 'shark', 'shrew',
                        'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                        'whale', 'willow_tree', 'wolf', 'woman', 'worm')


    def get_augment_transform_list(self):
        """ Gets list of training augmentation transforms
        """
        train_transform_list = [transforms.RandomCrop(32, padding=4)]
        train_transform_list += [transforms.RandomHorizontalFlip()]

        return train_transform_list


    def get_torchvision_dataset(self):
        """ """
        return torchvision.datasets.CIFAR100



class MNIST(TorchDataset):

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.num_classes = 10
        self.norm_mean = (0.1307,)
        self.norm_std  = (0.3081,)
        self.classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                        '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')


    def get_augment_transform_list(self):
        """ """
        return []


    def get_torchvision_dataset(self):
        """ """
        return torchvision.datasets.MNIST



class twoD(Dataset):
    """ This dataset can be fitted in memory and does not
    require torchvision.
    """

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.is_user_dataset = True
        self.num_classes = 2
        self.rect_side = 2
        self.test_grid = 0.01
        self.grid_arange = torch.arange(-1 + self.test_grid/2, 1, self.test_grid)


    def get_labels(self, inputs):
        """ """
        raise NotImplementedError


    def generate_set(self, nb_samples):
        """ """
        inputs = torch.empty(nb_samples, 2).uniform_(-self.rect_side/2, self.rect_side/2)
        labels = self.get_labels(inputs)

        return inputs, labels


    def get_test_set(self):
        """ """
        grid_x, grid_y = torch.meshgrid(self.grid_arange, self.grid_arange)

        inputs = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)
        labels = self.get_labels(inputs)

        return inputs, labels


    def init_plot_dict(self, mode='test'):
        """ """
        return {'inputs' : [], 'outputs' : []}


    def add_to_plot_dict(self, plot_dict, batch_data):
        """ plot_dict = {'inputs' : list, 'outputs': list} """

        plot_dict['inputs'].append(batch_data[0])
        plot_dict['outputs'].append(batch_data[1])


    def concatenate_plot_dict(self, plot_dict):
        """ Concatenate plot dictionary lists """

        inputs = torch.cat(plot_dict['inputs'], dim=0)
        outputs = torch.cat(plot_dict['outputs'], dim=0)

        return inputs, outputs


    def plot_train_imgs(self, inputs, labels):
        """ Plot train images (scatter plot)
        """
        check_device(inputs, labels, dev='cpu')

        inputs, labels = inputs.numpy(), labels.numpy()
        x, y = inputs[:, 0], inputs[:, 1]

        ax = plt.gca()
        self.add_gtruth_contour(ax, 'train')

        cm = plt.cm.get_cmap('RdYlBu')

        zero_mask = (labels < 0.5)
        plt.scatter(x[~zero_mask], y[~zero_mask], c=labels[~zero_mask],
                    vmin=0, vmax=1, s=40, cmap=cm)
        plt.scatter(x[zero_mask], y[zero_mask], c=labels[zero_mask],
                    vmin=0, vmax=1, s=8, cmap=cm)

        ax.set_aspect(aspect='equal')

        self.finalize_imgs(ax, 'train')


    def plot_test_imgs(self, inputs, probs):
        """ Plot output probability map
        """
        check_device(inputs, probs, dev='cpu')

        x_idx = (inputs[:, 0] + self.rect_side/2).div(self.test_grid).floor().to(torch.int64)
        y_idx = (self.rect_side/2 - inputs[:, 1]).div(self.test_grid).floor().to(torch.int64)

        z = torch.zeros((self.grid_arange.size(0), self.grid_arange.size(0)))
        z[y_idx, x_idx] = probs
        z = z.numpy() # probability value at (x, y) plot coordinates

        inputs = inputs.numpy()
        x, y = inputs[:, 0], inputs[:, 1] # (x, y) plot coordinates

        ax = plt.gca()
        self.add_gtruth_contour(ax, 'test')

        cm = plt.cm.get_cmap('GnBu')
        plt.imshow(z, vmin=0, vmax=1, cmap=cm, extent=(-1, 1, -1, 1), aspect='equal')

        self.finalize_imgs(ax, 'test')


    def finalize_imgs(self, ax, mode):
        """ """
        assert mode in ['train', 'test']

        cb = plt.colorbar(fraction=0.046, pad=0.04)

        lsize = 55
        cb.ax.tick_params(labelsize=lsize)
        ax.tick_params(axis='both', which='major', labelsize=lsize)

        plt.axis([-1, 1, -1, 1])

        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        fig = plt.gcf()
        img_size = 20
        fig.set_size_inches(img_size, img_size)

        if self.save_imgs:
            plt.savefig(os.path.join(self.log_dir_model, mode + '_img.pdf'), bbox_inches='tight')

        if self.plot_imgs:
            plt.show()

        plt.close()



class Circle(twoD):
    """ """

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.radius_sq = self.rect_side ** 2 / (2 * math.pi) # circle_area = rect_area/2


    def get_labels(self, inputs):
        """ """
        inputs_radius_sq = (inputs ** 2).sum(1)
        labels = (inputs_radius_sq < self.radius_sq).to(torch.float32)

        return labels


    def add_gtruth_contour(self, ax, mode):
        """ """
        assert mode in ['train', 'test']

        color = 'black' if mode == 'train' else 'chocolate'
        circle = plt.Circle((0, 0), radius=math.sqrt(self.radius_sq), color=color,
                                linewidth=3, fill=False)

        ax.add_artist(circle)



class S_shape(twoD):
    """ """

    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.base_sin = lambda t : 0.4*np.sin(-5*t)
        self.sin_shift = 0.3
        self.y_cutoff = 0.8


    def sin_func(self, t, type_):
        """ """
        assert type_ in ['upper', 'lower']

        if type_ == 'upper':
            return self.base_sin(t) + self.sin_shift
        else:
            return self.base_sin(t) - self.sin_shift


    def get_labels(self, inputs):
        """ """
        x, y = inputs[:, 0].numpy(), inputs[:, 1].numpy()

        in_sin = np.logical_and(x > self.sin_func(y, 'lower'), x < self.sin_func(y, 'upper')) # within the two sinusoids
        in_boundaries = (np.abs(y) < self.y_cutoff)  # within y boundaries

        np_labels = np.logical_and(in_sin, in_boundaries).astype(np.float32)

        return torch.from_numpy(np_labels)


    def add_gtruth_contour(self, ax, mode):
        """ """
        assert mode in ['train', 'test']

        color = 'black' if mode == 'train' else 'chocolate'

        a = 0.001
        t_sin = np.arange(-self.y_cutoff, self.y_cutoff + a, a)
        t_left = np.arange(self.sin_func(-self.y_cutoff, 'lower'), self.sin_func(-self.y_cutoff, 'upper') + a, a)
        t_right = np.arange(self.sin_func(self.y_cutoff, 'lower'), self.sin_func(self.y_cutoff, 'upper') + a, a)

        ax.plot(self.sin_func(t_sin, 'lower'), t_sin, c=color, linewidth=3)
        ax.plot(self.sin_func(t_sin, 'upper'), t_sin, c=color, linewidth=3)

        ax.plot(t_left, np.full_like(t_left, -self.y_cutoff),  c=color, linewidth=3)
        ax.plot(t_right, np.full_like(t_right, self.y_cutoff), c=color, linewidth=3)
