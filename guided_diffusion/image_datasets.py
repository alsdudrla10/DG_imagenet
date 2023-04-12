import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data_latent(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    partition=None,
    num_partition=10,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    print(data_dir)

    all_files = _list_image_files_recursively(data_dir)
    if partition != None:
        num_data = len(all_files)
        all_files = all_files[int(num_data*partition/num_partition):int(num_data*(partition+1)/num_partition)]
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset_latent(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        gen_data_path=None,
        real_data_path=None,
        config=None,

    ):
        super().__init__()
        self.config = config
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.gen_data_path = gen_data_path

        import pickle

        ##Gen data
        if config.data.dataset == 'IMAGENET64':
            self.gen_data = np.load(gen_data_path)["samples"][:config.classifier.num_data]
            self.gen_label = np.load(gen_data_path)["label"][:config.classifier.num_data]
        elif config.data.dataset == 'IMAGENET256' or config.data.dataset == 'IMAGENET256_manyclass':
            self.gen_data = np.load(gen_data_path)["arr_0"][:config.classifier.num_data]
            self.gen_label = np.load(gen_data_path)["arr_1"][:config.classifier.num_data]

        ##Real data
        if config.classifier.num_data == 1281167 or config.classifier.num_data == 250000 or config.classifier.num_data == 1300 or config.classifier.num_data == 14300:
            self.real_data = self.local_images
            self.real_labels = self.local_classes

            self.whole_label = np.concatenate([np.ones_like([1] * len(self.real_data)), np.zeros_like([1] * len(self.gen_data))])
            self.local_images = list(self.real_data) + list(self.gen_data)
            self.whole_condition = np.concatenate([np.array(self.real_labels), self.gen_label])

        else:
            with open(real_data_path, 'rb') as f:
                self.real_data = pickle.load(f)
                self.real_labels = np.array(pickle.load(f), dtype=np.int16)

            self.whole_label = np.concatenate([np.ones_like([1]*len(self.real_data)),np.zeros_like([1]*len(self.gen_data))])
            self.local_images =  list(self.real_data) + list(self.gen_data)
            self.whole_condition = np.concatenate([np.array(self.real_labels),self.gen_label])

        import random
        c = list(zip(self.local_images,self.whole_label,self.whole_condition))
        random.shuffle(c)
        self.local_images, self.whole_label, self.whole_condition = zip(*c)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        arr = self.local_images[idx]
        label = self.whole_label[idx]
        condition = self.whole_condition[idx]

        if label == 0.0:
            arr = self.local_images[idx]

        if label == 1.0:
            if self.config.classifier.num_data != 1281167 and self.config.classifier.num_data != 250000 and self.config.classifier.num_data != 1300 and self.config.classifier.num_data != 14300:
                arr = self.local_images[idx]
                pil_image = Image.fromarray(arr)
                if self.random_crop:
                    arr = random_crop_arr(pil_image, self.resolution)
                else:
                    arr = center_crop_arr(pil_image, self.resolution)
            else:
                path = self.local_images[idx]
                with bf.BlobFile(path, "rb") as f:
                    pil_image = Image.open(f)
                    pil_image.load()
                pil_image = pil_image.convert("RGB")
                if self.random_crop:
                    arr = random_crop_arr(pil_image, self.resolution)
                else:
                    arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        return [np.transpose(arr, [2, 0, 1]), condition], label


class ImageDataset_latent(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        arr = np.load(path)["samples"]
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32)

        y = np.load(path)["label"]
        return arr, y


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def no_crop(pil_image, image_size):
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    return arr

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
