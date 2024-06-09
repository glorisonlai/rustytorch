use super::{cpu, opencl};
use pyo3::prelude::*;

#[pyclass]
pub struct Tensor {
    #[pyo3(get)]
    pub data: Vec<f32>,
    #[pyo3(get)]
    pub strides: Vec<u32>,
    #[pyo3(get)]
    pub shape: Vec<u32>,
    #[pyo3(get)]
    pub ndim: u32,
    #[pyo3(get)]
    pub size: u32,
    #[pyo3(get)]
    pub device: String,
}

#[pymethods]
impl Tensor {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<u32>, ndim: u32, device: String) -> Tensor {
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
            .collect::<Vec<u32>>();

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

    pub fn to_device(&mut self, device: String) {
        match device.as_str() {
            "cpu" | "opencl" => {
                self.device = device;
            }
            _ => {
                panic!("Device not supported");
            }
        }
    }

    pub fn get(&self, index: Vec<u32>) -> f32 {
        let mut offset = 0;
        for i in 0..self.ndim {
            offset += index[i as usize] * self.strides[i as usize];
        }
        return self.data[offset as usize];
    }
}

#[pyfunction]
pub fn add_tensor(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
    if (tensor_a.ndim != tensor_b.ndim) || (tensor_a.shape != tensor_b.shape) {
        panic!("Tensor dimensions must match");
    }

    if tensor_a.device != tensor_b.device {
        panic!("Tensors must be on the same device");
    }

    let mut result_data = vec![0.0; tensor_a.size as usize];

    match tensor_a.device.as_str() {
        "cpu" => {
            let _ = cpu::add_tensor(tensor_a, tensor_b, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::add_tensor(tensor_a, tensor_b, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor_a.shape.clone(),
        tensor_a.ndim,
        tensor_a.device.clone(),
    );
}

#[pyfunction]
pub fn sub_tensor(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
    if (tensor_a.ndim != tensor_b.ndim) || (tensor_a.shape != tensor_b.shape) {
        panic!("Tensor dimensions must match");
    }

    if tensor_a.device != tensor_b.device {
        panic!("Tensors must be on the same device");
    }

    let mut result_data = vec![0.0; tensor_a.size as usize];

    match tensor_a.device.as_str() {
        "cpu" => {
            let _ = cpu::sub_tensor(tensor_a, tensor_b, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::sub_tensor(tensor_a, tensor_b, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor_a.shape.clone(),
        tensor_a.ndim,
        tensor_a.device.clone(),
    );
}

#[pyfunction]
pub fn mul_tensor(tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
    if (tensor_a.ndim != tensor_b.ndim) || (tensor_a.shape != tensor_b.shape) {
        panic!("Tensor dimensions must match");
    }

    if tensor_a.device != tensor_b.device {
        panic!("Tensors must be on the same device");
    }

    let mut result_data = vec![0.0; tensor_a.size as usize];

    match tensor_a.device.as_str() {
        "cpu" => {
            let _ = cpu::elementwise_mul_tensor(tensor_a, tensor_b, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::elementwise_mul_tensor(tensor_a, tensor_b, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor_a.shape.clone(),
        tensor_a.ndim,
        tensor_a.device.clone(),
    );
}

#[pyfunction]
pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Tensor {
    let mut result_data = vec![0.0; tensor.size as usize];

    match tensor.device.as_str() {
        "cpu" => {
            let _ = cpu::scalar_mul_tensor(tensor, scalar, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::scalar_mul_tensor(tensor, scalar, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor.shape.clone(),
        tensor.ndim,
        tensor.device.clone(),
    );
}

#[pyfunction]
pub fn sin_tensor(tensor: &Tensor) -> Tensor {
    let mut result_data = vec![0.0; tensor.size as usize];

    match tensor.device.as_str() {
        "cpu" => {
            let _ = cpu::sin_tensor(tensor, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::sin_tensor(tensor, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor.shape.clone(),
        tensor.ndim,
        tensor.device.clone(),
    );
}

#[pyfunction]
pub fn cos_tensor(tensor: &Tensor) -> Tensor {
    let mut result_data = vec![0.0; tensor.size as usize];

    match tensor.device.as_str() {
        "cpu" => {
            let _ = cpu::cos_tensor(tensor, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::cos_tensor(tensor, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor.shape.clone(),
        tensor.ndim,
        tensor.device.clone(),
    );
}

#[pyfunction]
pub fn reshape_tensor(tensor: &Tensor, shape: Vec<u32>, ndim: u32) -> Tensor {
    let size = shape.iter().fold(1, |acc, x| acc * x);

    if size != tensor.size {
        panic!("Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.");
    }

    let mut result_data = vec![0.0; size as usize];

    match tensor.device.as_str() {
        "cpu" => {
            let _ = cpu::assign_tensor(tensor, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::assign_tensor(tensor, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(result_data, shape, ndim, tensor.device.clone());
}

#[pyfunction]
pub fn zero_tensor(tensor: &Tensor) -> Tensor {
    let mut result_data = vec![0.0; tensor.size as usize];

    return Tensor::new(
        result_data,
        tensor.shape.clone(),
        tensor.ndim,
        tensor.device.clone(),
    );
}

#[pyfunction]
pub fn one_tensor(tensor: &Tensor) -> Tensor {
    let mut result_data = vec![0.0; tensor.size as usize];

    match tensor.device.as_str() {
        "cpu" => {
            let _ = cpu::fill_tensor(1.0, &mut result_data);
        }
        "opencl" => {
            let _ = opencl::fill_tensor(1.0, &mut result_data);
        }
        _ => {
            panic!("Device not supported");
        }
    }

    return Tensor::new(
        result_data,
        tensor.shape.clone(),
        tensor.ndim,
        tensor.device.clone(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensor() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let shape_a = vec![2, 2];
        let ndim_a = 2;
        let tensor_a = Tensor::new(data_a, shape_a, ndim_a, String::from("cpu"));

        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        let shape_b = vec![2, 2];
        let ndim_b = 2;
        let tensor_b = Tensor::new(data_b, shape_b, ndim_b, String::from("cpu"));

        let tensor_c = add_tensor(&tensor_a, &tensor_b);

        assert_eq!(tensor_c.get(vec![0, 0]), 6.0);
        assert_eq!(tensor_c.get(vec![0, 1]), 8.0);
        assert_eq!(tensor_c.get(vec![1, 0]), 10.0);
        assert_eq!(tensor_c.get(vec![1, 1]), 12.0);
    }

    #[test]
    #[should_panic]
    fn test_add_tensor_diff_dev() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape_a = vec![2, 2, 3];
        let ndim_a = 2;
        let tensor_a = Tensor::new(data_a, shape_a, ndim_a, "opencl".to_string());

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
