import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2 as transforms

'''

Standard API for all transforms: follow a different pattern compared to training preprocessing
where a "steps" dictionary was involved - here: 

Input:
    Tensor[C,H,W]
    ToImage()
        │
        ▼
    transform(img)
        │
        ▼
Output:
    Tensor[C,H,W]

'''


# *************************************************************************************** #
##### ---------------------------------     DEPTH    -----------------------------------  #
# *************************************************************************************** #

class DepthSplitTransform:

    def __init__(self, img_size):
        self.resize = v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        self._printed = False

    def __call__(self, img):

        img = img.float()

        img = torch.nan_to_num(
            img,
            nan=0.5,
            posinf=3.0,
            neginf=0.5,
        )

        img = img.clamp(0.5, 3.0)

        # optional resize
        # img = self.resize(img)

        ch1 = torch.where(img < 1.0, img - 0.5, 0.0)
        ch2 = torch.where((img >= 1.0) & (img < 2.0), img - 1.0, 0.0)
        ch3 = torch.where(img >= 2.0, img - 2.0, 0.0)

        img = torch.cat([ch1, ch2, ch3], dim=0)

        if not self._printed:
            self._printed = True
            print(img.shape, img.min(), img.max())

        return img

def depth_split_transform(cfg):
    return v2.Compose([
        v2.ToImage(),
        DepthSplitTransform(cfg.eval.img_size),
    ])

class DepthTransform:

    def __init__(self, img_size, single_channel, preprocessing, dynamic_depth, repeat_channels):
        self.resize = v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        self.repeat_channels = repeat_channels
        self.single_channel = single_channel
        self.preprocessing = preprocessing
        self.dynamic_depth = dynamic_depth



    def __call__(self, img):

        img = img.float()

        img = torch.nan_to_num(
            img,
            nan=0.5,
            posinf=3.0,
            neginf=0.5,
        )

        # img = img.clamp(0.5, 3.0)

        # img = self.resize(img)                # again leave out the resize

        if self.preprocessing == "minmax":
            # min-max normalization to [-1, 1]
            img = (img - 0.5) / (3.0 - 0.5)
            img = img * 2.0 - 1.0
        elif self.preprocessing == "stdmean":
            # N(0,1) normalization
            img = (img - 1.0485) / 0.4087
        else:
            raise ValueError(f"Unknown preprocessing method: {self.preprocessing}")

        # # TODO: previous normalization
        # img = (img - 0.5) / 2.5
        # img = img * 2.0 - 1.0

        ### TODO: new -> N(0,1) normalization for GT depth
        # img = (img - 1.0485) / 0.4087

        ### TODO: new -> N(0,1) normalization for DA3 single depth
        # img = (img - 0.9494) / 0.2244

        ### TODO: new -> N(0,1) normalization for DA3 triple depth
        # img = (img - 0.9034) / 0.2508

        if self.repeat_channels:
            img = img.repeat(3, 1, 1)

        if not hasattr(self, "_printed"):
            self._printed = True
            print("\n\n === DEBUG: AFTER TRANSFORM === :")
            print("shape:", img.shape)
            print("dtype:", img.dtype)
            print("min/max:", img.min().item(), img.max().item())



        # img = torch.zeros_like(img)                                 #### TODO: DEBUG ONLY - comment out
        # img = torch.rand_like(img)
        

        return img

def depth_only_transform(cfg, repeat_channels): 

    transform = DepthTransform(cfg.eval.img_size, cfg.single_channel, cfg.preprocessing, cfg.dynamic_depth, repeat_channels)

    return v2.Compose([
            v2.ToImage(),
            transform,
    ])



# *************************************************************************************** #
##### ---------------------------------     NORMALS    ---------------------------------- #
# *************************************************************************************** #


import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
import stable_pretraining as spt

class NormalMapTransform:
    """
    Convert uint8 normal maps to normalized unit vectors.

    Input:
        uint8 image (C,H,W) in [0,255]

    Output:
        float32 image (C,H,W) with unit normals in [-1,1]
    """

    def __init__(self, img_size, debug=False):
        self.resize = v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        self.debug = debug
        self._printed = False

    def __call__(self, img):

        if self.debug and not self._printed:
            print("\n===== BEFORE =====")
            print(img.shape, img.dtype)
            print("range:", img.min().item(), img.max().item())

        # uint8 -> float
        img = img.float()

        # [0,255] -> [-1,1]
        img = img / 127.5 - 1.0

        # resize
        img = self.resize(img)

        # restore unit-length normals
        img = F.normalize(img, dim=0, eps=1e-6)

        if self.debug and not self._printed:
            lengths = torch.linalg.norm(img, dim=0)

            print("\n===== AFTER =====")
            print(img.shape, img.dtype)
            print("range:", img.min().item(), img.max().item())
            print(
                f"normal lengths: "
                f"mean={lengths.mean():.4f} "
                f"std={lengths.std():.4f} "
                f"min={lengths.min():.4f} "
                f"max={lengths.max():.4f}"
            )
            print("=================\n")

            self._printed = True

        return img


def normal_transform(cfg):
    return v2.Compose(
        [
            v2.ToImage(),
            NormalMapTransform(
                img_size=cfg.eval.img_size,
                debug=True,   # disable after checking
            ),
        ]
    )

# *************************************************************************************** #
### ---------------------------------     RGBD     -------------------------------------- #
# *************************************************************************************** #


def img_transform(img_size):                         # --- RGB --- #
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            # transforms.Resize(img_size),                # resize not needed
        ]
    )
    return transform


class RGB_only_Transform:

    def __init__(self):
        self.normalize = transforms.Normalize(
            **spt.data.dataset_stats.ImageNet
        )

    def __call__(self, img):

        img = img.float()

        if img.max() > 1:
            img = img / 255.0

        # img = torch.zeros_like(img)                                       ### TODO: debug only - remove for eval

        return self.normalize(img)

def rgb_transform_call_debug_only(): 
    transform = RGB_only_Transform()
    return v2.Compose([
        v2.ToImage(),
        transform,
    ])


class RGBDTransform:
    """
    Apply the RGB and depth preprocessing independently and concatenate.

    Input:
        img: (4,H,W)
            RGB in channels 0:3
            Depth in channel 3

    Output:
        img: (4,H,W)
    """

    def __init__(self, img_size, debug=False):
        self.img_size = img_size
        self.rgb_transform = RGB_only_Transform()
        self.depth_transform = DepthTransform(self.img_size, single_channel=True, preprocessing="stdmean", repeat_channels=False, dynamic_depth=False)

        self.debug = debug
        self._printed = False

    def __call__(self, img):
        if self.debug and not self._printed:
            print("\n\n Debug inside RGBDTransform.__call__")
            print("Input:", img.shape, img.dtype)
            print(type(img))

            print("rgb-only stats before transform")
            rgb = img[:3]

            print(rgb.dtype)
            print(rgb.min().item(), rgb.max().item())

            rgb = rgb.float() / 255.

            print(rgb.min().item(), rgb.max().item())

        rgb = self.rgb_transform(img[:3])
        depth = self.depth_transform(img[3:])


        # OPTION 1: random permutation
        # depth = depth.flatten()                               ### TODO: commented-out: only for sanity checks (e.g. replace depth with random)
        # depth = depth[torch.randperm(depth.numel())]
        # depth = depth.view_as(depth)

        # OPTION 2: RGB mean 
        # depth = rgb.mean(dim=0, keepdim=True)

        # OPTION 3: replace with zeros everywhere
        # depth = torch.zeros_like(depth)

        out = torch.cat([rgb, depth], dim=0)

        if self.debug and not self._printed:
            print("\n===== RGBD =====")
            print("RGB:", rgb.shape, rgb.min().item(), rgb.max().item())
            print("Depth:", depth.shape, depth.min().item(), depth.max().item())
            print("Output:", out.shape)
            print("=================\n")
            self._printed = True

            print("\n\n === DEBUG ")
            depth = out[3]                     # [H,W]
            print(depth.min())
            print(depth.max())
            print(depth.mean())
            print(depth.std())

        return out


def rgbd_transform(cfg, debug=False):

    transform = RGBDTransform(cfg.eval.img_size, debug)

    return v2.Compose([
        v2.ToImage(),
        transform,
    ])


# *************************************************************************************** #
### ---------------------------------     POINT-MAPS     ---------------------------------# 
# *************************************************************************************** #


# OGBench point-map statistics (computed on the training split for the 1000 train + 100 val)
POINT_MEAN = torch.tensor([
    0.21534102,   # X
   -0.02312508,   # Y
    0.04208415,   # Z
], dtype=torch.float32)

POINT_STD = torch.tensor([
    0.43103240,   # X
    0.26829550,   # Y
    0.10406391,   # Z
], dtype=torch.float32)

POINT_MEAN_4D = POINT_MEAN.view(3, 1, 1)
POINT_STD_4D = POINT_STD.view(3, 1, 1)

class PointMapTransform(): 

    def __init__(self, debug=False):

        self.debug = debug
        self._printed = False

    def __call__(self, img):
        print("img.shape=", img.shape)

        # (3,H,W)
        points = img.float()

        mean = POINT_MEAN[:, None, None].to(points.device)      # [3,1,1]
        std  = POINT_STD[:, None, None].to(points.device)

        points = (points - mean) / std

        return points

def point_map_transform():
    transform = PointMapTransform()

    return v2.Compose([
        v2.ToImage(),
        transform,
    ])
    
