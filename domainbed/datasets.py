import os
import torch
import csv
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torchvision.transforms import ToPILImage, ToTensor, Resize, Normalize


from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

import numpy as np
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "MiniDomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",

]+[
    # multi-attribute MNIST
    "MNIST_ACause",
    "MNIST_AInd",
    "MNIST_ACauseUAInd",
    # small NORB
    "SmallNORB",
    "SmallNORB_ACause",
    "SmallNORB_AInd",
    "SmallNORB_ACauseUAInd",
    # Waterbirds
    "Waterbirds_ACause",
    "Waterbirds_Multiattr",
    # OoD bench
    'CelebA_Blond'
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        # Assuming 'original_images' is a torch.tensor object containing MNIST images
        print('original_images.shape: ', original_images.shape)
        # original_images = torch.stack(
        #     [self.transform_mnist_image(image) for image in original_images])
        original_images = self.transform_mnist_images(original_images)
        print('original_images.shape: ', original_images.shape)

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(
                images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def transform_mnist_images(self, images):
        # Convert tensor to PIL Images
        to_pil = ToPILImage()
        pil_images = [to_pil(image.squeeze()) for image in images]

        # Resize the images to (224, 224)
        resize = Resize((224, 224))
        resized_images = [resize(image) for image in pil_images]

        # Convert the images to RGB
        rgb_images = [image.convert('RGB') for image in resized_images]

        # Convert the RGB images to tensors and normalize
        to_tensor = ToTensor()
        normalize = Normalize((0.5,), (0.5,))
        transformed_images = [normalize(to_tensor(image))
                              for image in rgb_images]

        # Stack the transformed images into a single tensor
        stacked_images = torch.stack(transformed_images)

        return stacked_images
    # def transform_mnist_image(self, image):
    #     # Convert the tensor to a PIL Image
    #     image_pil = transforms.ToPILImage()(image)

    #     # Resize the image to (224, 224)
    #     resized_image = image_pil.resize((224, 224))

    #     # Convert the image to RGB
    #     rgb_image = Image.new("RGB", resized_image.size)
    #     rgb_image.paste(resized_image)

    #     # Normalize pixel values to the range of 0 to 1
    #     normalized_image = transforms.ToTensor()(rgb_image)

    #     return normalized_image


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        # self.input_shape = (2, 28, 28,)
        self.input_shape = (3, 224, 224)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

# single-attribute Causal


class MNIST_ACause(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = (0.1, 0.2, 0.9)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(
            images, labels, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        # ! remove environment, only return x, y
        # return TensorDataset(x, y, colors, colors)
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

# single-attribute Independent


class MNIST_AInd(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    # ENVIRONMENTS = ['+90%', '+80%', '-90%']
    ENVIRONMENTS = ['15', '60', '90']
    INPUT_SHAPE = (1, 14, 14)  # (3, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        # angles = ['15', '60', '90']
        for i, env in enumerate(self.ENVIRONMENTS[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(
                self.rotate_dataset(images, labels, self.ENVIRONMENTS[i]))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.rotate_dataset(
            images, labels, self.ENVIRONMENTS[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, int(angle), fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        x = torch.zeros(len(images), 1, 14, 14)
        for i in range(len(images)):
            x[i] = rotation(images[i].float().div_(255.0))

        y = labels.view(-1).long()

        return TensorDataset(x, y, y, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

# multi-attribute Causal + Independent


class MNIST_ACauseUAInd(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)

        original_images = original_dataset_tr.train_data
        original_labels = original_dataset_tr.train_labels

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = (0.1, 0.2, 0.9)
        angles = ['15', '60', '90']
        for i, env in enumerate(environments[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(
                images, labels, env, angles[i]))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(
            images, labels, environments[-1], angles[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment, angle):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # rotate the image by angle in parameter
        images = self.rotate_dataset(images, angle)
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images  # .float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y, colors, colors)

    def rotate_dataset(self, images, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: transforms.functional.rotate(
                x, int(angle), fill=(0,))),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 14, 14)
        for i in range(len(images)):
            x[i] = rotation(images[i].float().div_(255.0))
        return x

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class MiniDomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["clipart", "painting", "real", "sketch"]
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "MiniDomainNet/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3",
                    "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


# Waterbirds dataset
class Waterbirds(torch.utils.data.Dataset):
    def __init__(self, data_dir, root_images, y_array, confounder_array, transform=None, train=True, augment=False):
        self.data_dir = data_dir
        self.root_images = root_images
        self.y_array = y_array
        self.confounder_array = confounder_array
        self.transform = transform
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):

        y = self.y_array[idx]
        bgd = self.confounder_array[idx]
        img_filename = os.path.join(
            self.data_dir,
            self.root_images[idx])
        img = Image.open(img_filename).convert('RGB')

        # Apply weather augmentation
        add_effect_flag = 0
        if self.augment:
            img = np.array(img)
            if self.train:
                add_effect_flag = np.random.choice([0, 1])
                if add_effect_flag == 1:
                    img = am.darken(img, darkness_coeff=0.5)
            else:
                img = am.add_rain(img, rain_type='heavy', slant=20)
            img = Image.fromarray(img)

        # Apply transform
        img = self.transform(img)
        x = img

        return x, y
        # return x, y, bgd, add_effect_flag


class Waterbirds_ACause(MultipleDomainDataset):

    ENVIRONMENTS = ["tr1", "tr2", "val", "test1", "test2", "test3", "test4"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.data_dir = os.path.join(
            root,
            # 'dataset',
            '_'.join(['waterbird_complete95'] + ['forest2water2']))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2

        # Get the y values
        self.y_array = self.metadata_df['y'].values

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) +
                            self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.train_transform = get_transform_cub(
            train=True,
        )
        self.eval_transform = get_transform_cub(
            train=False,
        )

        self.datasets = []

        # get train subset
        split = 'train'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        train_filename_array = self.filename_array[indices]
        train_y_array = self.y_array[indices]
        train_group_array = self.group_array[indices]
        train_confounder_array = self.confounder_array[indices]

        # train domains based on |A|
        for group_idx in range(self.n_confounders+1):
            group_mask = train_confounder_array == group_idx
            group_indices = np.where(group_mask)[0]
            self.datasets.append(Waterbirds(self.data_dir,
                                            train_filename_array[group_indices],
                                            train_y_array[group_indices],
                                            train_confounder_array[group_indices],
                                            self.train_transform))

        # get val subset
        split = 'val'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        val_filename_array = self.filename_array[indices]
        val_y_array = self.y_array[indices]
        val_group_array = self.group_array[indices]
        val_confounder_array = self.confounder_array[indices]

        self.datasets.append(Waterbirds(self.data_dir,
                                        val_filename_array,
                                        val_y_array,
                                        val_confounder_array,
                                        self.eval_transform))

        # get test subset
        split = 'test'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        test_filename_array = self.filename_array[indices]
        test_y_array = self.y_array[indices]
        test_group_array = self.group_array[indices]
        test_confounder_array = self.confounder_array[indices]

        # test domains based on |A| x |Y|
        for group_idx in range(self.n_groups):
            group_mask = test_group_array == group_idx
            group_indices = np.where(group_mask)[0]
            self.datasets.append(Waterbirds(self.data_dir,
                                            test_filename_array[group_indices],
                                            test_y_array[group_indices],
                                            test_confounder_array[group_indices],
                                            self.eval_transform))


class Waterbirds_Multiattr(MultipleDomainDataset):
    ENVIRONMENTS = ["tr", "val", "test1", "test2", "test3", "test4"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.data_dir = os.path.join(
            root,
            # 'dataset',
            '_'.join(['waterbird_complete95'] + ['forest2water2']))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2

        # Get the y values
        self.y_array = self.metadata_df['y'].values

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) +
                            self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.train_transform = get_transform_cub(
            train=True,
        )
        self.eval_transform = get_transform_cub(
            train=False,
        )

        self.datasets = []

        # get train subset
        split = 'train'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        train_filename_array = self.filename_array[indices]
        train_y_array = self.y_array[indices]
        train_group_array = self.group_array[indices]
        train_confounder_array = self.confounder_array[indices]

        self.datasets.append(Waterbirds(self.data_dir,
                                        train_filename_array,
                                        train_y_array,
                                        train_confounder_array,
                                        self.train_transform))

        # get val subset
        split = 'val'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        val_filename_array = self.filename_array[indices]
        val_y_array = self.y_array[indices]
        val_group_array = self.group_array[indices]
        val_confounder_array = self.confounder_array[indices]

        self.datasets.append(Waterbirds(self.data_dir,
                                        val_filename_array,
                                        val_y_array,
                                        val_confounder_array,
                                        self.eval_transform))

        # get test subset
        split = 'test'
        mask = self.split_array == self.split_dict[split]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        test_filename_array = self.filename_array[indices]
        test_y_array = self.y_array[indices]
        test_group_array = self.group_array[indices]
        test_confounder_array = self.confounder_array[indices]

        # test domains based on |A| x |Y|
        for group_idx in range(self.n_groups):
            group_mask = test_group_array == group_idx
            group_indices = np.where(group_mask)[0]
            self.datasets.append(Waterbirds(self.data_dir,
                                            test_filename_array[group_indices],
                                            test_y_array[group_indices],
                                            test_confounder_array[group_indices],
                                            self.eval_transform))


def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = (224, 224)

    if (not train):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class CelebA_Environment(Dataset):
    def __init__(self, target_attribute_id, split_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        file_names = []
        attributes = []
        with open(split_csv) as f:
            reader = csv.reader(f)
            next(reader)  # discard header
            for row in reader:
                file_names.append(row[0])
                attributes.append(np.array(row[1:], dtype=int))
        attributes = np.stack(attributes, axis=0)
        self.samples = list(zip(file_names, list(
            attributes[:, target_attribute_id])))
        # Find unique values in the specified column and convert to a list
        unique_values = np.unique(attributes[:, target_attribute_id]).tolist()
        # Map the unique values to the corresponding labels
        self.classes = ['not blond' if value ==
                        0 else 'blond' for value in unique_values]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, label = self.samples[index]
        image = Image.open(Path(self.img_dir, file_name))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


class CelebA_Blond(MultipleDomainDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['tr_env1', 'tr_env2', 'te_env']
    TARGET_ATTRIBUTE_ID = 9

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if 'data_augmentation_scheme' in hparams:
            raise NotImplementedError(
                'CelebA_Blond has its own data augmentation scheme')

        transform = transforms.Compose([
            # crop the face at the center, no stretching
            transforms.CenterCrop(178),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            get_normalize(),
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0),
                                         ratio=(1.0, 1.3333333333333333)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),  # do not alter hue
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_normalize(),
        ])

        img_dir = Path(root, 'celeba', 'img_align_celeba')
        self.datasets = []
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            split_csv = Path(root, 'celeba', 'blond_split', f'{env_name}.csv')
            dataset = CelebA_Environment(self.TARGET_ATTRIBUTE_ID, split_csv, img_dir,
                                         env_transform)
            self.datasets.append(dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2  # blond or not
