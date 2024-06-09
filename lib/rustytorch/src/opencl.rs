extern crate ocl;
use super::tensor;
use ocl::{Buffer, MemFlags, ProQue};

fn create_buffer(data: &Vec<f32>, pro_que: &ProQue) -> ocl::Result<Buffer<f32>> {
    Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(data.len())
        .copy_host_slice(data)
        .build()
}

pub fn add_tensor(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_add(__global float* vec_a, __global float* vec_b, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = vec_a[i] + vec_b[i];
        }
    "#;

    let dim_size = tensor_a.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_a_buffer = create_buffer(&tensor_a.data, &pro_que)?;

    let data_b_buffer = create_buffer(&tensor_b.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_add")
        .arg(&data_a_buffer)
        .arg(&data_b_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn sub_tensor(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_sub(__global float* vec_a, __global float* vec_b, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = vec_a[i] - vec_b[i];
        }
    "#;

    let dim_size = tensor_a.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_a_buffer = create_buffer(&tensor_a.data, &pro_que)?;

    let data_b_buffer = create_buffer(&tensor_b.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_sub")
        .arg(&data_a_buffer)
        .arg(&data_b_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn elementwise_mul_tensor(
    tensor_a: &tensor::Tensor,
    tensor_b: &tensor::Tensor,
    result_data: &mut Vec<f32>,
) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_mul(__global float* vec_a, __global float* vec_b, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = vec_a[i] * vec_b[i];
        }
    "#;

    let dim_size = tensor_a.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_a_buffer = create_buffer(&tensor_a.data, &pro_que)?;

    let data_b_buffer = create_buffer(&tensor_b.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_mul")
        .arg(&data_a_buffer)
        .arg(&data_b_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn scalar_mul_tensor(
    tensor: &tensor::Tensor,
    scalar: f32,
    result_data: &mut Vec<f32>,
) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_fill(__global float* vec, float scalar, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = scalar * vec[i];
        }
    "#;

    let dim_size = tensor.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_buffer = create_buffer(&tensor.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_mul")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn sin_tensor(tensor: &tensor::Tensor, result_data: &mut Vec<f32>) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_fill(__global float* vec, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = sinf(vec[i]);
        }
    "#;

    let dim_size = tensor.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_buffer = create_buffer(&tensor.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_mul")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn cos_tensor(tensor: &tensor::Tensor, result_data: &mut Vec<f32>) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_fill(__global float* vec, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = cosf(vec[i]);
        }
    "#;

    let dim_size = tensor.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_buffer = create_buffer(&tensor.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_mul")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn assign_tensor(tensor: &tensor::Tensor, result_data: &mut Vec<f32>) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_assign(__global float* vec, __global float* vec_res) {
            int i = get_global_id(0);

            vec_res[i] = vec[i];
        }
    "#;

    let dim_size = tensor.size;

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let data_buffer = create_buffer(&tensor.data, &pro_que)?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_assign")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

pub fn fill_tensor(scalar: f32, result_data: &mut Vec<f32>) -> ocl::Result<()> {
    let kernel = r#"
        __kernel void vector_fill(float scalar, __global float* vec_res) {
            vec_res[get_global_id(0)] = scalar;
        }
    "#;

    let dim_size = result_data.len();

    let pro_que = ProQue::builder().src(kernel).dims(dim_size).build()?;

    let res_buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("vector_fill")
        .arg(scalar)
        .arg(&res_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    res_buffer.read(result_data).enq()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vector() {
        let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "opencl".to_string());
        let tensor_b = tensor::Tensor::new(vec![4.0, 5.0, 6.0], vec![3], 1, "opencl".to_string());

        let mut result_data = vec![0.0; 3];

        let _ = add_tensor(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_vector() {
        let tensor_a = tensor::Tensor::new(vec![6.0, 5.0, 4.0], vec![3], 1, "opencl".to_string());
        let tensor_b = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "opencl".to_string());

        let mut result_data = vec![0.0; 3];

        let _ = sub_tensor(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![5.0, 3.0, 1.0]);
    }

    #[test]
    fn test_mul_vector() {
        let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "opencl".to_string());
        let tensor_b = tensor::Tensor::new(vec![4.0, 5.0, 6.0], vec![3], 1, "opencl".to_string());

        let mut result_data = vec![0.0; 3];

        let _ = elementwise_mul_tensor(&tensor_a, &tensor_b, &mut result_data);

        assert_eq!(result_data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_assign_vector() {
        let tensor_a = tensor::Tensor::new(vec![1.0, 2.0, 3.0], vec![3], 1, "opencl".to_string());

        let mut result_data = vec![0.0; 3];

        let _ = assign_tensor(&tensor_a, &mut result_data);

        assert_eq!(result_data, vec![1.0, 2.0, 3.0]);
    }
}
