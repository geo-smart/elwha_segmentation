import numpy as np
import os
import cv2
from pathlib import Path
from typing import Any, Callable, Optional
from torchvision.datasets.vision import VisionDataset

def initializeFolders():
  folders = [
    "../data/img/",
    "../out/",
    "../out/intermediate/",
    "../out/realignment/",
    "../out/geotiff/",
  ]

  for folder in folders:
    if not os.path.exists(folder):
      os.makedirs(folder)

def getSavePath(type: str, number: int) -> str:
  if (type == "RGB"):
    return f"../data/img/airborne_{number + 1}.png"
  if (type == "IR"):
    return f"../data/img/airborne_ir_{number}.png"
  
  if (type == "STITCH"):
    return f"../out/intermediate/stitch_{number + 1}.png"
  if (type == "MATCH"):
    return f"../out/intermediate/match_{number + 1}.png"
  if (type == "KEYPOINTS"):
    return f"../out/intermediate/keypoints_{number + 1}.png"
  if (type == "MASK"):
    return f"../out/intermediate/mask_{number + 1}.png"

  if (type == "GEOTIFF"):
    return f"../out/geotiff/tiff_{number}.tiff"
  if (type == "GEOSAM"):
    return f"../out/geotiff/predict_{number}.tiff"
  
  if (type == "FINAL"):
    return f"../out/realignment/rgb_{number + 1}.png"
  else:
    raise ValueError(f"{type} is not a recognized save type!")

def percentage(loss, total):
  if (hasattr(loss, '__len__')):
    return round(100 * len(loss) / total, 2)
  else:
    return round(100 * loss / total, 2)
  
def distance(n, m):
  return abs(np.sum(n - m))

# Used for cv2.decomposeHomographyMat, which returns the 
# arrays inside single element tuples for some reason?
def unTuple(d):
  _, r, t, n = d
  return r[0], t[0], n[0]

def isNearCenter(point, dimensions, threshold):
  dx_center = abs(point[0] - (dimensions[1] / 2)) / dimensions[1]
  dy_center = abs(point[1] - (dimensions[0] / 2)) / dimensions[0]

  return (dx_center < threshold) and (dy_center < threshold)

def getQuadrant(point, dimensions):
  left = True if point[0] < (dimensions[1] / 2) else False
  top = True if point[1] < (dimensions[0] / 2) else False

  if (top and left): return 0
  if (top and not left): return 1
  if (not top and not left): return 2
  if (not top and left): return 3

def withinThresholdDeviations(value, threshold, baseline):
  return value < baseline + threshold and value > baseline - threshold

def onlyNumeric(seq):
  seq = str(seq)
  seq_type= type(seq)
  return seq_type().join(filter(seq_type.isdigit, seq))

def getPaths(directory: Path, glob: str):
  paths = np.array(sorted(directory.glob(glob), key=onlyNumeric))
  print(f"Found and loaded {len(paths)} images with glob {glob}.")
  return paths

def writeSetToFile(file_path: str, target_set: set):
  with open(file_path, "w") as file:
    for element in target_set:
      file.write(str(element) + "\n")

def readSetFromFile(file_path: str, data_type = int):
  loaded_set = set()
  with open(file_path, "r") as file:
    for line in file:
      element = data_type(line.strip())
      loaded_set.add(element)
  return loaded_set

class SupervisedDataset(VisionDataset):
  """
  Adapted from Manpreet Singh Minhas's transfer learning
  for semantic segmentation tutorial.

  A PyTorch dataset for image segmentation task, intended for supervised learning tasks 
  where ground truth data needs to be supplied. The dataset is also compatible with 
  torchvision transforms. The transforms passed would be applied to both images and masks.
  """
  def __init__(
      self,
      root: str,
      ir_folder: str,
      rgb_folder: str,
      masks_folder: str,

      ir_prefix: str,
      rgb_prefix: str,
      masks_prefix: str,

      transforms: Optional[Callable] = None,
      seed: int = None, # type: ignore
      fraction: float = None, # type: ignore
      subset: str = None, # type: ignore
    ) -> None:
    """
    ARGS:
      root (str): Used for super init of VisionDataset.
      ir_folder (str): Folder containing the ir images.
      rgb_folder (str): Folder containing the rgb images.
      masks_folder (str): Folder containing the segmentation truth mask images.
      ir_prefix (str): Prefix string for ir images, e.g. "ir_image_" if stored as ir_image_1, ir_image_2 ...
      rgb_prefix (str): Prefix string for the rgb images.
      masks_prefix (str): Prefix string for the segmentation truth mask images.

      transforms (Optional[Callable], optional): A function/transform that takes in a sample and returns a transformed version. E.g, ``transforms.ToTensor`` for images.
      seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
      fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
      subset (str, optional): "Train" or "Test" to select the appropriate set. Defaults to None.
    
    RAISES:
      OSError: If any of ir_folder, rgb_folder or masks_folder do not exist.
      ValueError: If subset is not either "Train" or "Test".
    """
    super().__init__(root, transforms)

    ir_images_path = Path(self.root) / ir_folder
    rgb_images_path = Path(self.root) / rgb_folder
    mask_images_path = Path(self.root) / masks_folder

    if not rgb_images_path.exists():
      raise OSError(f"{rgb_images_path} does not exist.")
    if not mask_images_path.exists():
      raise OSError(f"{mask_images_path} does not exist.")

    if not fraction:
      self.mask_names = sorted(mask_images_path.glob(masks_prefix + "*.png"))
    else:
      if subset not in ["Train", "Test"]:
        raise ValueError(f"{subset} is not a valid input. Acceptable values are Train and Test.")
      
      self.fraction = fraction
      self.mask_list = getPaths(mask_images_path, masks_prefix + "*.png")
      
      # We don't have an rgb reconstruction for every ir image
      # and definitely don't have a mask for every rgb + ir pair.
      # We only want to grab ir and rgb images we have truth data for.

      self.ir_image_list = []
      self.rgb_image_list = []
      for mask in self.mask_list:
        number = onlyNumeric(mask)
        self.ir_image_list.append(ir_images_path / (ir_prefix + number + ".png"))
        self.rgb_image_list.append(rgb_images_path / (rgb_prefix + number + ".png"))

      self.ir_image_list = np.array(self.ir_image_list)
      self.rgb_image_list = np.array(self.rgb_image_list)
      assert len(self.ir_image_list) == len(self.rgb_image_list) == len(self.mask_list)

      if seed:
        np.random.seed(seed)
        indices = np.arange(len(self.mask_list))
        np.random.shuffle(indices)

        self.ir_image_list = self.ir_image_list[indices]
        self.rgb_image_list = self.rgb_image_list[indices]
        self.mask_list = self.mask_list[indices]


      if subset == "Train":
        self.ir_image_names = self.ir_image_list[:int(np.ceil(len(self.ir_image_list) * (1 - self.fraction)))]
        self.rgb_image_names = self.rgb_image_list[:int(np.ceil(len(self.ir_image_list) * (1 - self.fraction)))]
        self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.fraction)))]
      else:
        self.ir_image_names = self.ir_image_list[int(np.ceil(len(self.ir_image_list) * (1 - self.fraction))):]
        self.rgb_image_names = self.rgb_image_list[int(np.ceil(len(self.rgb_image_list) * (1 - self.fraction))):]
        self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]
        
      print(f"Subset of {len(self.mask_names)} ground truth segmentation masks marked for {subset}.")

  def __len__(self) -> int:
    return len(self.mask_names)
  
  def __getitem__(self, index: int) -> Any:
    ir_image_path: Path = self.ir_image_names[index]
    rgb_image_path: Path = self.rgb_image_names[index]
    mask_path: Path = self.mask_names[index]

    # ir_image = im.open(ir_image_file)
    # ir_image = ir_image.convert("L")
    ir_image = cv2.imread(str(ir_image_path), cv2.IMREAD_GRAYSCALE)
    ir_predict = cv2.cvtColor(cv2.imread(str(ir_image_path)), cv2.COLOR_BGR2RGB) # SAM can only predict on RGB images, not grayscale

    # rgb_image = im.open(rgb_image_file)
    # rgb_image = rgb_image.convert("RGB")
    rgb_image = cv2.imread(str(rgb_image_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path))[:,:,0]
    display_mask = cv2.bitwise_not(mask).astype(np.int32)
    
    sample = {
      "ir": ir_image, "rgb": rgb_image, "mask": mask, "display_mask": display_mask,
      "ir_path": ir_image_path, "rgb_path": rgb_image_path, "mask_path": mask_path,
      "predict": ir_predict,
    }

    if self.transforms:
      sample["ir"] = self.transforms(sample["ir"]) # type: ignore
      sample["rgb"] = self.transforms(sample["rgb"]) # type: ignore
      sample["mask"] = self.transforms(sample["mask"]) # type: ignore
    return sample

class UnsupervisedDataset(VisionDataset):
  """
  A PyTorch dataset for image segmentation tasks, intended for
  unsupervised tasks with no ground truth data.
  """
  def __init__(
      self,
      root: str,
      ir_folder: str,
      rgb_folder: str,
      ir_prefix: str,
      rgb_prefix: str,
      seed: int = None, # type: ignore
    ) -> None:
    """
    ARGS:
      root (str): Used for super init of VisionDataset.
      ir_folder (str): Folder containing the ir images.
      rgb_folder (str): Folder containing the rgb images.
      ir_prefix (str): Prefix string for ir images, e.g. "ir_image_" if stored as ir_image_1, ir_image_2 ...
      rgb_prefix (str): Prefix string for the rgb images.
      seed (int, optional): Specify a seed for the train and test split for reproducible results.
    
    RAISES:
      OSError: If any of ir_folder, rgb_folder or masks_folder do not exist.
      ValueError: If subset is not either "Train" or "Test".
    """
    super().__init__(root, None)

    ir_images_path = Path(self.root) / ir_folder
    rgb_images_path = Path(self.root) / rgb_folder

    if not rgb_images_path.exists():
      raise OSError(f"{rgb_images_path} does not exist.")

    # ir_image_names = getPaths(ir_images_path, ir_prefix + "*.png")
    rgb_image_names = getPaths(rgb_images_path, rgb_prefix + "*.png")
    self.length = len(rgb_image_names)

    ir_image_names = []
    for rgb_image in rgb_image_names:
      number = onlyNumeric(rgb_image)
      ir_image_names.append(ir_images_path / (ir_prefix + number + ".png"))
    ir_image_names = np.array(ir_image_names)

    if seed:
      np.random.seed(seed)
      indices = np.arange(self.length)
      np.random.shuffle(indices)

      self.ir_image_names = ir_image_names[indices]
      self.rgb_image_names = rgb_image_names[indices]
    else:
      self.ir_image_names = ir_image_names
      self.rgb_image_names = rgb_image_names

  def __len__(self) -> int:
    return self.length
  
  def __getitem__(self, index: int) -> Any:
    ir_image_path: Path = self.ir_image_names[index]
    rgb_image_path: Path = self.rgb_image_names[index]

    ir_image = cv2.imread(str(ir_image_path), cv2.IMREAD_GRAYSCALE)
    ir_predict = cv2.cvtColor(cv2.imread(str(ir_image_path)), cv2.COLOR_BGR2RGB) # SAM can only predict on RGB images, not grayscale
    
    rgb_image = cv2.imread(str(rgb_image_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    sample = {
      "ir": ir_image, "rgb": rgb_image, "predict": ir_predict,
      "ir_path": ir_image_path, "rgb_path": rgb_image_path,
    }
    return sample

class ImageDataset():
  """
  A utility class for managing a large set of images for a segmentation task, intended for 
  supervised learning tasks where ground truth data needs to be supplied.
  """
  def __init__(
      self,
      images_folder: str,
      masks_folder: str,
      masks_glob: str,
      include_masks: list = None,
      exclude_masks: list = None,
      subset: str = None,
      fraction: float = None,
    ):
    images_path = Path(images_folder)
    masks_path = Path(masks_folder)

    if not images_path.exists():
      raise OSError(f"{images_path} does not exist.")
    if not masks_path.exists():
      raise OSError(f"{masks_path} does not exist.")
    if subset not in ["Train", "Test"]:
      raise ValueError(f"{subset} is not a valid input. Acceptable values are Train and Test.")
    
    self.mask_list = getPaths(masks_path, masks_glob)
    masks_suffix = masks_glob.replace("*", "").replace(".png", "")
    identifier = lambda path: path.stem.replace(masks_suffix, "")

    if include_masks is not None:
      original_amnt = len(self.mask_list)
      self.mask_list = [mask for mask in self.mask_list if identifier(mask) in include_masks]
      print(f"Pruned {original_amnt - len(self.mask_list)} masks based on set of {len(exclude_masks)} included masks.")
    
    if exclude_masks is not None: 
      original_amnt = len(self.mask_list)
      self.mask_list = [mask for mask in self.mask_list if identifier(mask) not in exclude_masks]
      print(f"Pruned {original_amnt - len(self.mask_list)} masks from set of {len(exclude_masks)} excluded masks.")
    
    self.names_list = []
    self.image_list = []
    
    for mask in self.mask_list:
      name = identifier(mask)
      self.names_list.append(name)

      image = images_path / (name + mask.suffix)
      self.image_list.append(image)

      if not image.exists():
        raise OSError(f"Image '{image}' does not exist!")

    self.names_list = np.array(self.names_list)
    self.image_list = np.array(self.image_list)
    self.mask_list = np.array(self.mask_list)
    assert len(self.image_list) == len(self.mask_list) == len(self.names_list)

    self.fraction = fraction
    cutoff = int(np.ceil(len(self.mask_list) * (1 - self.fraction)))

    if subset == "Train":
      self.names = self.names_list[:cutoff]
      self.image_names = self.image_list[:cutoff]
      self.mask_names = self.mask_list[:cutoff]
    else:
      self.names = self.names_list[cutoff:]
      self.image_names = self.image_list[cutoff:]
      self.mask_names = self.mask_list[cutoff:]
      
    print(f"Subset of {len(self.mask_names)} ground truth segmentation masks marked for {subset}.")

  def __len__(self) -> int:
    return len(self.mask_names)
  
  def __getitem__(self, index: int) -> Any:
    image_path: Path = self.image_names[index]
    mask_path: Path = self.mask_names[index]

    try:
      image = cv2.imread(str(image_path))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(str(mask_path))[:,:,0]
    except:
      print(f"Failed to read image '{self.names[index]}'")
    
    sample = { "name": self.names[index], "image": image, "mask": mask }
    return sample