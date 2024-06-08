use super::cpu;
use pyo3::prelude::*;

#[pyclass]
pub struct Tensor {
    #[pyo3(get)]
    pub data: Vec<f32>,
    #[pyo3(get)]
    pub strides: Vec<usize>,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub ndim: usize,
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub device: String,
}

#[pymethods]
impl Tensor {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>, ndim: usize, device: String) -> Tensor {
        let size = shape.iter().fold(1, |acc, x| acc * x);

        let strides = shape
            .iter()
            .rev()
            .fold(Vec::from([1]), |mut acc, x| {
                if acc.len() != shape.len() {
                    acc.push(acc[acc.len() - 1] * x);
                }
                acc
            })
            .into_iter()
            .rev()
            .collect::<Vec<usize>>();

        let tensor = Tensor {
            data,
            strides,
            shape,
            ndim,
            size,
            device,
        };

        return tensor;
    }

    pub fn get(&self, index: Vec<usize>) -> f32 {
        let mut offset = 0;
        for i in 0..self.ndim {
            offset += index[i] * self.strides[i];
        }
        return self.data[offset as usize];
    }
}

#[pyfunction]
pub fn add_tensor(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
    if (tensor_a.ndim != tensor_b.ndim) || (tensor_a.shape != tensor_b.shape) {
        panic!("Tensor dimensions must match");
    }

    let mut result_data = vec![0.0; tensor_a.size];

    let _ = cpu::add_tensor_cpu(tensor_a, tensor_b, &mut result_data);

    return Tensor::new(
        result_data,
        tensor_a.shape.clone(),
        tensor_a.ndim,
        tensor_a.device.clone(),
    );
}

#[pyfunction]
pub fn reshape_tensor(tensor: &Tensor, shape: Vec<usize>, ndim: usize) -> Tensor {
    let size = shape.iter().fold(1, |acc, x| acc * x);

    if size != tensor.size {
        panic!("Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.");
    }

    let mut result_data = vec![0.0; size];

    let _ = cpu::assign_tensor_cpu(tensor, &mut result_data);

    return Tensor::new(result_data, shape, ndim, tensor.device.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensor() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = vec![2, 2];
        let ndim_a = 2;
        let tensor_a = Tensor::new(data_a, shape_a, ndim_a);

        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let shape_b = vec![2, 2];
        let ndim_b = 2;
        let tensor_b = Tensor::new(data_b, shape_b, ndim_b);

        let tensor_c = add_tensor(&tensor_a, &tensor_b);

        assert_eq!(tensor_c.get(vec![0, 0]), 6.0);
        assert_eq!(tensor_c.get(vec![0, 1]), 8.0);
        assert_eq!(tensor_c.get(vec![1, 0]), 10.0);
        assert_eq!(tensor_c.get(vec![1, 1]), 12.0);
    }

    #[test]
    #[should_panic]
    fn test_bad_add_tensor() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape_a = vec![2, 2, 3];
        let ndim_a = 2;
        let tensor_a = Tensor::new(data_a, shape_a, ndim_a, "cpu".to_string());

        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let shape_b = vec![2, 2, 3, 4];
        let ndim_b = 2;
        let tensor_b = Tensor::new(data_b, shape_b, ndim_b, "cpu".to_string());

        let _ = add_tensor(&tensor_a, &tensor_b);
    }

    #[test]
    fn test_assign_tensor() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = vec![2, 2];
        let ndim_a = 2;
        let tensor_a = Tensor::new(data_a, shape_a, ndim_a, "cpu".to_string());

        let shape_b = vec![4];
        let ndim_b = 1;
        let tensor_b = reshape_tensor(&tensor_a, shape_b, ndim_b);

        assert_eq!(tensor_b.get(vec![0]), 1.0);
        assert_eq!(tensor_b.get(vec![1]), 2.0);
        assert_eq!(tensor_b.get(vec![2]), 3.0);
        assert_eq!(tensor_b.get(vec![3]), 4.0);
    }
}
