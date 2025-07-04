import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
from .folder import default_loader
from .vision import VisionDataset
except ImportError:
    # Fallback for direct execution
    from PIL import Image
    
    def default_loader(path):
        """Default image loader using PIL."""
        return Image.open(path).convert('RGB')
    
    class VisionDataset:
        """Base class for vision datasets."""
        def __init__(self, root, transform=None, target_transform=None):
            self.root = root
            self.transform = transform
            self.target_transform = target_transform


class Flickr8kParser(HTMLParser):
    """Parser for extracting captions from the Flickr8k dataset web page."""

    def __init__(self, root: Union[str, Path]) -> None:
        super().__init__()

        self.root = root

        # Data structure to store captions
        self.annotations: Dict[str, List[str]] = {}

        # State variables
        self.in_table = False
        self.current_tag: Optional[str] = None
        self.current_img: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.current_tag = tag

        if tag == "table":
            self.in_table = True

    def handle_endtag(self, tag: str) -> None:
        self.current_tag = None

        if tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_table:
            if data == "Image Not Found":
                self.current_img = None
            elif self.current_tag == "a":
                img_id = data.split("/")[-2]
                img_id = os.path.join(self.root, img_id + "_*.jpg")
                img_id = glob.glob(img_id)[0]
                self.current_img = img_id
                self.annotations[img_id] = []
            elif self.current_tag == "li" and self.current_img:
                img_id = self.current_img
                self.annotations[img_id].append(data.strip())


class Flickr8k(VisionDataset):
    """`Flickr8k Entities <http://hockenmaier.cs.illinois.edu/8k-pictures.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    def __init__(
        self,
        root: Union[str, Path],
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        parser = Flickr8kParser(self.root)
        with open(self.ann_file) as fh:
            parser.feed(fh.read())
        self.annotations = parser.annotations

        self.ids = list(sorted(self.annotations.keys()))
        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        img = self.loader(img_id)
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class Flickr30k(VisionDataset):
    """`Flickr30k Entities <https://bryanplummer.com/Flickr30kEntities/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file) as fh:
            for line in fh:
                img_id, caption = line.strip().split("\t")
                self.annotations[img_id[:-2]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))
        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

def download_flickr8k(root_dir="./data"):
    """Download Flickr8k dataset."""
    import urllib.request
    import zipfile
    
    os.makedirs(root_dir, exist_ok=True)
    
    # URLs for Flickr8k dataset
    urls = {
        "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "text": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(root_dir, f"Flickr8k_{name}.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            print(f"Extracted {name}")
    
    return root_dir

def main():
    """Main function to test dataset loading."""
    print("Testing Flickr8k dataset loading...")
    
    # Download dataset
    root_dir = download_flickr8k()
    
    # Test dataset loading
    try:
        # Typical paths after extraction
        image_root = os.path.join(root_dir, "Flicker8k_Dataset")
        ann_file = os.path.join(root_dir, "Flickr8k.token.txt")
        
        if os.path.exists(image_root) and os.path.exists(ann_file):
            dataset = Flickr8k(root=image_root, ann_file=ann_file)
            print(f"Dataset loaded successfully! Total images: {len(dataset)}")
            
            # Test loading first item
            if len(dataset) > 0:
                img, captions = dataset[0]
                print(f"First image shape: {img.size if hasattr(img, 'size') else 'N/A'}")
                print(f"Number of captions: {len(captions)}")
                print(f"First caption: {captions[0] if captions else 'No captions'}")
        else:
            print("Dataset files not found after download.")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()