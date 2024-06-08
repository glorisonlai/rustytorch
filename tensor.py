import pyto2rch
from typing import TypeAlias, Union, TypeVar

type NestedList[T] = list[Union[T, 'NestedList[T]']]

class Tensor:
    ndim: int
    shape: list[int]
    tensor: pyto2rch.Tensor

    def __init__(self, data = None, device = "cpu") -> None:
        if not data is None:
            flat_data, flat_shape = self.flatten(data)
            ndim = len(flat_shape)
            self.ndim = ndim
            self.shape = flat_shape
            self.tensor = pyto2rch.Tensor(flat_data, flat_shape, ndim, device)
        else:
            self.tensor = None
            self.ndim = 0
            self.shape = []

    def flatten(self, nested_list: NestedList[float]) -> tuple[list[float], list[int]]:
        def _aux_flatten(nested_list: float | NestedList[float]) -> tuple[list[float], list[int]]:
            flat_data = []
            shape = []
            if isinstance(nested_list, list):
                inner_shape = []
                for sublist in nested_list:
                    inner_data, inner_shape = _aux_flatten(sublist)
                    flat_data.extend(inner_data)
                shape.append(len(nested_list))
                shape.extend(inner_shape)
            else:
                flat_data.append(nested_list)
            return flat_data, shape
        flat_data, shape = _aux_flatten(nested_list)
        return flat_data, shape 
    
    def __getitem__(self, indices: tuple[int, ...]) -> float:
        """
        Access the tensor element at the given indices [i, j, k...]
        """
        if len(indices) != self.ndim:
            raise IndexError("Invalid number of indices")

        return self.tensor.get(indices)

    def reshape(self, new_shape: list[int]) -> 'Tensor':
        """
        Reshape the tensor to the given shape
        """
        result_data = Tensor()
        ndim = len(new_shape)
        result_data.tensor = pyto2rch.reshape_tensor(self.tensor, new_shape, ndim)
        result_data.ndim = ndim
        result_data.shape = new_shape

        return result_data

    def __add__(self, other: 'Tensor') -> 'Tensor':
        """
        Add two tensors element-wise
        """
        if other.shape != self.shape:
            raise ValueError("Tensors must have the same shape")

        result_data = Tensor()
        result_data.tensor = pyto2rch.add_tensor(self.tensor, other.tensor)
        result_data.ndim = self.ndim
        result_data.shape = self.shape

        return result_data

if __name__ == "__main__":
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    t3 = t1 + t2
    print(t3[1,1])
