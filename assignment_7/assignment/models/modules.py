import operator

import torch


class Sum(torch.nn.Module):
    # Based on parts of torch.nn.ModuleList
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(*list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def forward(self, input):
        result = self[0](input)
        for module in self[1:]:
            result += module(input)
        return result


class Product(torch.nn.Module):
    # Based on parts of torch.nn.ModuleList
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(*list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def forward(self, input):
        result = self[0](input)
        for module in self[1:]:
            result *= module(input)
        return result


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.shape = shape

    def forward(self, input):
        output = torch.reshape(input, self.shape)
        return output


class Normalize(torch.nn.Module):
    def __init__(self, p_norm=2, dim=-1):
        super().__init__()

        self.dim = dim
        self.p_norm = p_norm

    def forward(self, input):
        output = torch.nn.functional.normalize(input, p=self.p_norm, dim=self.dim)
        return output
