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
        result_data[i] = tensor_a.data[i] + tensor_b.data[i];
    }

    Ok(())
}

fn sub_tensor_cpu(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> TensorResult<()> {
    for i in 0..tensor_a.size {
        result_data[i] = tensor_a.data[i] - tensor_b.data[i];
    }

    Ok(())
}

fn elementwise_mul_tensor_cpu(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> TensorResult<()> {
    for i in 0..tensor_a.size {
        result_data[i] = tensor_a.data[i] * tensor_b.data[i];
    }

    Ok(())
}

pub fn assign_tensor_cpu(tensor: &tensor::Tensor, result_data: &mut Vec<f32>) -> TensorResult<()> {
    for i in 0..tensor.size {
        result_data[i] = tensor.data[i];
    }

    Ok(())
}
