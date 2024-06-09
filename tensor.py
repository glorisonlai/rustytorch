from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from custom_types import NestedList
import rustytorch 
from autograd import AddBackward, BackwardFunction

class Tensor:
    ndim: int
    shape: list[int]
    device: str
    tensor: rustytorch.Tensor
    requires_grad: bool
    grad: 'Tensor | None'
    grad_fn: BackwardFunction 

    def __init__(self, data = None, device = "cpu") -> None:
        if not data is None:
            flat_data, flat_shape = self.flatten(data)
            ndim = len(flat_shape)
            self.ndim = ndim
            self.shape = flat_shape
            self.device = device
            self.tensor = rustytorch.Tensor(flat_data, flat_shape, ndim, device)
            self.requires_grad = False 
            self.grad = None
            self.grad_fn = None
        else:
            self.tensor = None
            self.ndim = 0
            self.device = device
            self.shape = []
            self.requires_grad = True 
            self.grad = None
            self.grad_fn = None

    def flatten(self, nested_list: 'NestedList[float]') -> tuple[list[float], list[int]]:
        def _aux_flatten(nested_list: 'float | NestedList[float]') -> tuple[list[float], list[int]]:
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

    def sin(self) -> 'Tensor':
        res_tensor = self.tensor.sin_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = res_tensor
        result_data.ndim = self.ndim
        result_data.shape = self.shape
        result_data.device = self.device

        return result_data

    def cos(self) -> 'Tensor':
        res_tensor = self.tensor.cos_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = res_tensor
        result_data.ndim = self.ndim
        result_data.shape = self.shape
        result_data.device = self.device

        return result_data

    def zero_like(self) -> 'Tensor':
        res_tensor = rustytorch.zero_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = res_tensor
        result_data.ndim = self.ndim
        result_data.shape = self.shape
        result_data.device = self.device

        return result_data

    def ones_like(self) -> 'Tensor':
        res_tensor = rustytorch.one_tensor(self.tensor)

        result_data = Tensor()
        result_data.tensor = res_tensor
        result_data.ndim = self.ndim
        result_data.shape = self.shape
        result_data.device = self.device

        return result_data
    
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
        result_data.tensor = rustytorch.reshape_tensor(self.tensor, new_shape, ndim)
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
        result_data.tensor = rustytorch.add_tensor(self.tensor, other.tensor)
        result_data.ndim = self.ndim
        result_data.shape = self.shape
        result_data.requires_grad = self.requires_grad or other.requires_grad

        if self.requires_grad:
            result_data.grad_fn = AddBackward(self, other)

        return result_data

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """
        Subtracts two tensors element-wise
        """
        if other.shape != self.shape:
            raise ValueError("Tensors must have the same shape")

        result_data = Tensor()
        result_data.tensor = rustytorch.sub_tensor(self.tensor, other.tensor)
        result_data.ndim = self.ndim
        result_data.shape = self.shape

        return result_data

    def __mul__(self, other: 'Tensor | int | float') -> 'Tensor':
        if isinstance(other, Tensor):
            return self.mul_tensor(other)
        elif isinstance(other, (float, int)):
            return self.mul_scalar(other)
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

    def mul_tensor(self, other: 'Tensor') -> 'Tensor':
        """
        Multiplies two tensors element-wise
        """
        if other.shape != self.shape:
            raise ValueError("Tensors must have the same shape")

        result_data = Tensor()
        result_data.tensor = rustytorch.mul_tensor(self.tensor, other.tensor)
        result_data.ndim = self.ndim
        result_data.shape = self.shape

        return result_data

    def mul_scalar(self, scalar: int | float) -> 'Tensor':
        """
        Multiplies the tensor by a scalar
        """
        result_data = Tensor()
        result_data.tensor = rustytorch.mul_scalar(self.tensor, scalar)
        result_data.ndim = self.ndim
        result_data.shape = self.shape

        return result_data

    def __rmul__(self, other: 'Tensor') -> 'Tensor':
        return self.__mul__(other)

    def __neg__(self) -> 'Tensor':
        return self.__mul__(-1)

    def __pos__(self) -> 'Tensor':
        return self

    def to(self, device: str) -> 'Tensor':
        """
        Move the tensor to the given device

        Returns itself
        """
        self.device = device
        self.tensor.to_device(device)

        return self 

    def zero_grad(self):
        self.grad = None

    def detach(self):
        self.grad = None
        self.grad_fn = None

    def backward(self, gradient: 'Tensor') -> None:
        if not self.requires_grad:
            return
        if gradient is None:
            if self.shape == [1]:
                grad = Tensor([1.0])
            else: 
                raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")

        stack = [(self, gradient)]
        visited = set()
    
        while stack:
            tensor, grad = stack.pop()
            
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

            # Propagate gradients to inputs if not a leaf tensor
            if tensor.grad_fn is not None:
                grads = tensor.grad_fn.backward(grad)
                for tensor, grad in zip(tensor.grad_fn.input, grads):
                    if isinstance(tensor, Tensor) and tensor not in visited:
                        stack.append((tensor, grad))
                        visited.add(tensor)



if __name__ == "__main__":
    t1 = Tensor([[1, 2], [3, 4]]).to("opencl")
    t2 = Tensor([[5, 6], [7, 8]], "opencl")
    t3 = t1 + t2
    print(t3[1,1])
