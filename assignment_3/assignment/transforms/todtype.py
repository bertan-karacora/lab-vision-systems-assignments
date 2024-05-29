import torchvision.transforms.v2 as tv_transforms

import assignment.libs.utils_import as utils_import


class ToDtype(tv_transforms.Transform):
    def __init__(self, dtype, scale=False):
        super().__init__()

        self.dtype = dtype
        self.scale = scale
        self.transform_tv = None

        self._initialize()

    def _initialize(self):
        self.transform_tv = tv_transforms.ToDtype(
            dtype=utils_import.import_dtype(self.dtype),
            scale=self.scale,
        )

    def _transform(self, features, params):
        features_transformed = self.transform_tv._transform(features, params)
        return features_transformed
