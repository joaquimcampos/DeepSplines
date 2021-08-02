import os
import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractproperty, abstractmethod

from deepsplines.ds_utils import check_device, denormalize, init_sub_dir


def init_dataset(**params):
    """
    Initialize dataset.

    Returns:
        dataset (Dataset object).
    """

    # add your datasets here and create a corresponding Dataset class
    dataset_dict = {
        's_shape': S_shape,
        'circle': Circle,
        'cifar10': Cifar10,
        'cifar100': Cifar100,
        'mnist': MNIST
    }

    if params['dataset_name'] not in dataset_dict.keys():
        raise ValueError('Chosen dataset is not available... Exiting.')

    dataset = dataset_dict[params['dataset_name']](**params)

    return dataset


def generate_save_dataset(dataset_name,
                          data_dir,
                          num_train_samples=1500,
                          num_valid_samples=1500):
    """
    Args:
        dataset_name (str): 's_shape' or 'circle'
        data_dir (str): Data directory
        num_train_samples (int)
        num_valid_samples (int)
    """
    if not os.path.isdir(data_dir):
        print(f'\nData directory {data_dir} not found. Creating it.')
        os.makedirs(data_dir)

    dataset_dir = init_sub_dir(data_dir, dataset_name)

    params = {
        'dataset_name': dataset_name,
        'log_dir': dataset_dir,
        'plot_imgs': False,
        'save_imgs': True
    }

    dataset = init_dataset(**params)

    print(f'\nSaving {dataset_name} dataset in {dataset_dir}')

    for mode in ['train', 'valid']:
        num_samples = (num_train_samples
                       if mode == 'train' else num_valid_samples)
        inputs, labels = dataset.generate_set(num_samples)

        if mode == 'train':
            dataset.plot_train_imgs(inputs, labels)  # save training images

        save_dict = {'inputs': inputs, 'labels': labels}
        torch.save(save_dict,
                   os.path.join(dataset.log_dir_model, mode + '_data.pth'))

    inputs, labels = dataset.get_test_set()
    dataset.plot_test_imgs(inputs, labels)  # save test images

    save_dict = {'inputs': inputs, 'labels': labels}
    torch.save(save_dict, os.path.join(dataset.log_dir_model, 'test_data.pth'))


class Dataset(ABC):
    """ Abstract class for datasets """
    def __init__(self,
                 dataset_name=None,
                 log_dir=None,
                 model_name=None,
                 plot_imgs=False,
                 save_imgs=False,
                 **kwargs):
        """
        Args:
            dataset_name (str):
                s_shape', 'circle', 'cifar10', 'cifar100' or 'mnist'.
            log_dir (str):
                log_directory for saving images if model_name is None.
            model_name (str):
                If given, the log_directory for saving images is
                'log_dir/model_name/'.
        """

        self.name = dataset_name

        if model_name is not None:
            self.log_dir_model = os.path.join(log_dir, model_name)
        else:
            self.log_dir_model = log_dir

        self.plot_imgs = plot_imgs
        self.save_imgs = save_imgs
        self.get_plot = plot_imgs or save_imgs

    @abstractproperty
    def is_user_dataset(self):
        """ True if the dataset is user-defined (bool) """
        pass

    @abstractproperty
    def num_classes(self):
        """ Number of classes in the dataset (int) """
        pass

    @abstractmethod
    def plot_train_imgs(self, *args, **kwargs):
        """ Plots/saves train images """
        pass

    @property
    def get_test_imgs(self):
        """ True if plotting/saving validation/test images (bool) """

        return self.get_plot and self.is_user_dataset


class TorchDataset(Dataset):
    """ Abstract class for Torchvision datasets """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self._is_user_dataset = False  # torchvision dataset
        self.sample_batch = 4  # show 4 sample images

    @property
    def is_user_dataset(self):
        """ """
        return self._is_user_dataset

    @abstractproperty
    def norm_mean(self):
        """ Mean for each channel in the training dataset (tuple). """
        pass

    @abstractproperty
    def norm_std(self):
        """
        Standard deviation for each channel in the training
        dataset (tuple).
        """
        pass

    @abstractproperty
    def classes(self):
        """ Names of the classes (tuple). """
        pass

    def get_minimal_transform_list(self):
        """
        Get minimal dataset transforms.

        Applied to train, validation and test sets.
        """
        normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        minimal_transform_list = [transforms.ToTensor(), normalize]

        return minimal_transform_list

    @abstractmethod
    def get_augment_transform_list(self):
        """
        Get training dataset augmentation transforms.

        This is dataset-dependent.
        """
        pass

    def get_train_valid_transforms(self):
        """ Get training and validation dataset transforms. """

        # augment training data
        train_transform_list = self.get_augment_transform_list()
        minimal_transform_list = self.get_minimal_transform_list()

        train_transform = transforms.Compose(train_transform_list +
                                             minimal_transform_list)
        valid_transform = transforms.Compose(minimal_transform_list)

        return train_transform, valid_transform

    def get_test_transform(self):
        """ Get training and validation dataset transforms. """

        minimal_transform_list = self.get_minimal_transform_list()
        test_transform = transforms.Compose(minimal_transform_list)

        return test_transform

    @abstractmethod
    def get_torchvision_dataset(self):
        """ Returns a torchvision dataset """
        pass

    def plot_train_imgs(self, trainloader):
        """
        Plots a sample batch of training images/labels from a
        torchvision dataset.

        Args:
            trainloader (iter):
                iterator of input images-label pairs.
        """
        data_iter = iter(trainloader)
        images, labels = data_iter.next()  # get first batch
        images = denormalize(images, self.norm_mean, self.norm_std)\
            .clamp(min=0, max=1)

        self.plot_torch_dataset_samples(images[0:self.sample_batch],
                                        labels[0:self.sample_batch])

    def plot_torch_dataset_samples(self,
                                   images,
                                   true_labels,
                                   pred_labels=None):
        """
        Plot images/labels from a torchvision dataset.

        Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/

        Args:
            images (np.array).
            true_labels (list):
                list of true labels.
            pred_labels (list):
                list of predicted labels.
        """
        assert math.sqrt(images.size(0)) % 1 == 0, \
            'plot_images expects a batch size which is a perfect power'

        H = int(math.sqrt(images.size(0)))
        # convert to numpy format (N x H x W x C)
        images = images.numpy().transpose([0, 2, 3, 1])
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
                xlabel = "True: {0}\nPred: {1}".format(true_labels_name,
                                                       pred_labels_name)
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()


class Cifar10(TorchDataset):
    """ Class for CIFAR10 Dataset """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self._num_classes = 10
        self._norm_mean = (0.4914, 0.4822, 0.4465)
        self._norm_std = (0.2470, 0.2435, 0.2616)
        self._classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck')

    @property
    def num_classes(self):
        """ """
        return self._num_classes

    @property
    def norm_mean(self):
        """ """
        return self._norm_mean

    @property
    def norm_std(self):
        """ """
        return self._norm_std

    @property
    def classes(self):
        """ """
        return self._classes

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
    """ Class for CIFAR100 Dataset """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self._num_classes = 100
        self._norm_mean = (0.5072, 0.4867, 0.4412)
        self._norm_std = (0.2673, 0.2564, 0.2762)
        self._classes = (
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
            'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
            'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
            'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
            'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
            'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
            'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
            'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
            'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
            'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm')

    @property
    def num_classes(self):
        """ """
        return self._num_classes

    @property
    def norm_mean(self):
        """ """
        return self._norm_mean

    @property
    def norm_std(self):
        """ """
        return self._norm_std

    @property
    def classes(self):
        """ """
        return self._classes

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
        self._num_classes = 10
        self._norm_mean = (0.1307, )
        self._norm_std = (0.3081, )
        self._classes = ('0 - zero', '1 - one', '2 - two', '3 - three',
                         '4 - four', '5 - five', '6 - six', '7 - seven',
                         '8 - eight', '9 - nine')

    @property
    def num_classes(self):
        """ """
        return self._num_classes

    @property
    def norm_mean(self):
        """ """
        return self._norm_mean

    @property
    def norm_std(self):
        """ """
        return self._norm_std

    @property
    def classes(self):
        """ """
        return self._classes

    def get_augment_transform_list(self):
        """ """
        return []

    def get_torchvision_dataset(self):
        """ """
        return torchvision.datasets.MNIST


class twoD(Dataset):
    """
    Abstract class for 2D datasets that can be fitted in memory and do not
    require torchvision.
    """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self._is_user_dataset = True
        self._num_classes = 2
        self.rect_side = 2
        self.test_grid = 0.01
        self.grid_arange = torch.arange(-1 + self.test_grid / 2, 1,
                                        self.test_grid)

    @property
    def is_user_dataset(self):
        """ """
        return self._is_user_dataset

    @property
    def num_classes(self):
        """ """
        return self._num_classes

    @abstractmethod
    def get_labels(self, inputs):
        """
        Generate dataset labels for a set of inputs.

        Args:
            inputs (torch.Tensor).
        Returns:
            labels (torch.Tensor).
        """
        pass

    def generate_set(self, nb_samples):
        """
        Generate training or validation dataset.

        Args:
            num_samples (int).
        """
        inputs = torch.empty(nb_samples, 2)\
            .uniform_(-self.rect_side / 2, self.rect_side / 2)

        labels = self.get_labels(inputs)

        return inputs, labels

    def get_test_set(self):
        """ Generate test dataset """

        grid_x, grid_y = torch.meshgrid(self.grid_arange, self.grid_arange)

        inputs = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)),
                           dim=1)
        labels = self.get_labels(inputs)

        return inputs, labels

    def init_plot_dict(self, mode='test'):
        """ Init plot dictionary """

        return {'inputs': [], 'outputs': []}

    def add_to_plot_dict(self, plot_dict, batch_data):
        """ plot_dict = {'inputs' : list, 'outputs': list} """

        plot_dict['inputs'].append(batch_data[0])
        plot_dict['outputs'].append(batch_data[1])

    def concatenate_plot_dict(self, plot_dict):
        """ Concatenate plot dictionary lists """

        inputs = torch.cat(plot_dict['inputs'], dim=0)
        outputs = torch.cat(plot_dict['outputs'], dim=0)

        return inputs, outputs

    @abstractmethod
    def add_gtruth_contour(self, ax, mode):
        """
        Add contour of gtruth to plot.

        Args:
            ax (matplotlib.axes):
                plot axes.
            mode (str):
                'train' or 'test' (determines color).
        """
        pass

    def plot_train_imgs(self, inputs, labels):
        """ Plot training images (scatter plot) """

        check_device(inputs, labels, dev='cpu')

        inputs, labels = inputs.numpy(), labels.numpy()
        x, y = inputs[:, 0], inputs[:, 1]

        ax = plt.gca()
        self.add_gtruth_contour(ax, 'train')

        cm = plt.cm.get_cmap('RdYlBu')

        zero_mask = (labels < 0.5)
        plt.scatter(x[~zero_mask],
                    y[~zero_mask],
                    c=labels[~zero_mask],
                    vmin=0,
                    vmax=1,
                    s=40,
                    cmap=cm)
        plt.scatter(x[zero_mask],
                    y[zero_mask],
                    c=labels[zero_mask],
                    vmin=0,
                    vmax=1,
                    s=8,
                    cmap=cm)

        ax.set_aspect(aspect='equal')

        self.finalize_imgs(ax, 'train')

    def plot_test_imgs(self, inputs, probs):
        """ Plot output test probability map """

        check_device(inputs, probs, dev='cpu')

        x_idx = (inputs[:, 0] + self.rect_side / 2)\
            .div(self.test_grid).floor().to(torch.int64)

        y_idx = (self.rect_side / 2 - inputs[:, 1])\
            .div(self.test_grid).floor().to(torch.int64)

        z = torch.zeros((self.grid_arange.size(0), self.grid_arange.size(0)))
        z[y_idx, x_idx] = probs
        z = z.numpy()  # probability value at (x, y) plot coordinates

        ax = plt.gca()
        self.add_gtruth_contour(ax, 'test')

        cm = plt.cm.get_cmap('GnBu')
        plt.imshow(z,
                   vmin=0,
                   vmax=1,
                   cmap=cm,
                   extent=(-1, 1, -1, 1),
                   aspect='equal')

        self.finalize_imgs(ax, 'test')

    def finalize_imgs(self, ax, mode):
        """ Finalize image plotting """
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
            plt.savefig(os.path.join(self.log_dir_model, mode + '_img.pdf'),
                        bbox_inches='tight')

        if self.plot_imgs:
            plt.show()

        plt.close()


class Circle(twoD):
    """ Class for a 2D circle dataset """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        # circle_area = rect_area/2
        self.radius_sq = self.rect_side**2 / (2 * math.pi)

    def get_labels(self, inputs):
        """
        Generate dataset labels for a set of inputs.

        Args:
            inputs (torch.Tensor).
        Returns:
            labels (torch.Tensor).
        """
        inputs_radius_sq = (inputs**2).sum(1)
        labels = (inputs_radius_sq < self.radius_sq).to(torch.float32)

        return labels

    def add_gtruth_contour(self, ax, mode):
        """
        Add contour of circle to plot.

        Args:
            ax (matplotlib.axes):
                plot axes.
            mode (str):
                'train' or 'test' (determines color).
        """
        assert mode in ['train', 'test']

        color = 'black' if mode == 'train' else 'chocolate'
        circle = plt.Circle((0, 0),
                            radius=math.sqrt(self.radius_sq),
                            color=color,
                            linewidth=3,
                            fill=False)

        ax.add_artist(circle)


class S_shape(twoD):
    """ Class for a 2D circle dataset """
    def __init__(self, **params):
        """ """
        super().__init__(**params)
        self.base_sin = lambda t: 0.4 * np.sin(-5 * t)
        self.sin_shift = 0.3
        self.y_cutoff = 0.8

    def sin_func(self, t, type_):
        """ Shifted sin_func for boundaries of s_shape.

        Args:
            t (torch.Tensor): input values.
            type_ (str): 'upper' or 'lower'.
        """
        assert type_ in ['upper', 'lower']

        if type_ == 'upper':
            return self.base_sin(t) + self.sin_shift
        else:
            return self.base_sin(t) - self.sin_shift

    def get_labels(self, inputs):
        """
        Generate dataset labels for a set of inputs.

        Args:
            inputs (torch.Tensor).
        Returns:
            labels (torch.Tensor).
        """
        x, y = inputs[:, 0].numpy(), inputs[:, 1].numpy()

        # within the two sinusoids
        in_sin = np.logical_and(x > self.sin_func(y, 'lower'),
                                x < self.sin_func(y, 'upper'))
        in_boundaries = (np.abs(y) < self.y_cutoff)  # within y boundaries

        np_labels = np.logical_and(in_sin, in_boundaries).astype(np.float32)

        return torch.from_numpy(np_labels)

    def add_gtruth_contour(self, ax, mode):
        """
        Add contour of s-shape to plot.

        Args:
            ax (matplotlib.axes):
                plot axes.
            mode (str):
                'train' or 'test' (determines color).
        """
        assert mode in ['train', 'test']

        color = 'black' if mode == 'train' else 'chocolate'

        a = 0.001
        t_sin = np.arange(-self.y_cutoff, self.y_cutoff + a, a)
        t_left = np.arange(self.sin_func(-self.y_cutoff, 'lower'),
                           self.sin_func(-self.y_cutoff, 'upper') + a, a)
        t_right = np.arange(self.sin_func(self.y_cutoff, 'lower'),
                            self.sin_func(self.y_cutoff, 'upper') + a, a)

        ax.plot(self.sin_func(t_sin, 'lower'), t_sin, c=color, linewidth=3)
        ax.plot(self.sin_func(t_sin, 'upper'), t_sin, c=color, linewidth=3)

        ax.plot(t_left,
                np.full_like(t_left, -self.y_cutoff),
                c=color,
                linewidth=3)
        ax.plot(t_right,
                np.full_like(t_right, self.y_cutoff),
                c=color,
                linewidth=3)
