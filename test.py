import torch
import ctypes

print(torch.rand([5, 4, 8]).stride())

t = torch.rand([5, 4, 8])

print(t.shape)
# [5, 4, 8]

print(t.stride())
# [32, 8, 1]

new_t = t.reshape([4, 5, 2, 2, 2])

print(new_t.shape)
# [4, 5, 2, 2, 2]

print(new_t.stride())
# [40, 8, 4, 2, 1]

def print_internal(t: torch.Tensor):
    print(
        torch.frombuffer(
            ctypes.string_at(t.data_ptr(), t.storage().nbytes()), dtype=t.dtype
        )
    )

print_internal(t)
# [0.0752, 0.5898, 0.3930, 0.9577, 0.2276, 0.9786, 0.1009, 0.138, ...

print_internal(new_t)

t = torch.arange(0, 24).reshape(2, 3, 4)
print(t)
# [[[ 0,  1,  2,  3],
#   [ 4,  5,  6,  7],
#   [ 8,  9, 10, 11]],
 
#  [[12, 13, 14, 15],
#   [16, 17, 18, 19],
#   [20, 21, 22, 23]]]

print(t.shape)
# [2, 3, 4]

print(t.stride())
# [12, 4, 1]

new_t = t.transpose(0, 1)
print(new_t)
# [[[ 0,  1,  2,  3],
#   [12, 13, 14, 15]],

#  [[ 4,  5,  6,  7],
#   [16, 17, 18, 19]],

#  [[ 8,  9, 10, 11],
#   [20, 21, 22, 23]]]

print(new_t.shape)
# [3, 2, 4]

print(new_t.stride())
# [4, 12, 1]

new_t_contiguous = new_t.contiguous()

print(new_t_contiguous.is_contiguous())

print(new_t)
# [[[ 0,  1,  2,  3],
#   [12, 13, 14, 15]],

#  [[ 4,  5,  6,  7],
#   [16, 17, 18, 19]],

#  [[ 8,  9, 10, 11],
#   [20, 21, 22, 23]]]

print_internal(new_t)
# [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

print_internal(new_t_contiguous)
# [ 0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7, 16, 17, 18, 19,  8,  9, 10, 11, 20, 21, 22, 23]
