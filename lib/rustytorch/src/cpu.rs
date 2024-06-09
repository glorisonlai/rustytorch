use super::tensor;
use std::fmt;

type TensorResult<T> = std::result::Result<T, TensorError>;

// Define our error types. These may be customized for our error handling cases.
// Now we will be able to write our own errors, defer to an underlying error
// implementation, or do something in between.
#[derive(Debug, Clone)]
pub struct TensorError;

// Generation of an error is completely separate from how it is displayed.
// There's no need to be concerned about cluttering complex logic with the display style.
//
// Note that we don't store any extra info about the errors. This means we can't state
// which string failed to parse without modifying our types to carry that information.
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid tensor operation")
    }
}

pub fn add_tensor_cpu(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> TensorResult<()> {
    for i in 0..tensor_a.size {
        result_data[i as usize] = tensor_a.data[i as usize] + tensor_b.data[i as usize];
    }

    Ok(())
}

pub fn sub_tensor_cpu(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> TensorResult<()> {
    for i in 0..tensor_a.size {
        result_data[i as usize] = tensor_a.data[i as usize] - tensor_b.data[i as usize];
    }

    Ok(())
}

pub fn elementwise_mul_tensor_cpu(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> TensorResult<()> {
    for i in 0..tensor_a.size {
        result_data[i as usize] = tensor_a.data[i as usize] * tensor_b.data[i as usize];
    }

    Ok(())
}

pub fn assign_tensor_cpu(tensor: &tensor::Tensor, result_data: &mut Vec<f32>) -> TensorResult<()> {
    for i in 0..tensor.size {
        result_data[i as usize] = tensor.data[i as usize];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensor_cpu() {
        let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "cpu".to_string());
        let tensor_b = tensor::Tensor::new(vec![4.0, 5.0, 6.0], vec![3], 1, "cpu".to_string());

        let mut result_data = vec![0.0; tensor_a.size as usize];

        let _ = super::add_tensor_cpu(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_tensor_cpu() {
        let tensor_a = tensor::Tensor::new(vec![6.0, 5.0, 4.0], vec![3], 1, "cpu".to_string());
        let tensor_b = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "cpu".to_string());

        let mut result_data = vec![0.0; tensor_a.size as usize];

        let _ = super::sub_tensor_cpu(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![5.0, 3.0, 1.0]);
    }

    #[test]
    fn test_mul_tensor_cpu() {
        let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "cpu".to_string());
        let tensor_b = tensor::Tensor::new(vec![4.0, 5.0, 6.0], vec![3], 1, "cpu".to_string());

        let mut result_data = vec![0.0; tensor_a.size as usize];

        let _ = super::elementwise_mul_tensor_cpu(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_assign_tensor_cpu() {
        let tensor = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "cpu".to_string());

        let mut result_data = vec![0.0; tensor.size as usize];

        let _ = super::assign_tensor_cpu(&tensor, &mut result_data);

        assert_eq!(result_data, vec![1.0, 2.0, 3.0]);
    }
}
