
#[cfg(test)]
mod tests {
    use std::error::Error;

    use matchbox::Tensor;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn tensor_info() {
        let a = Tensor::new(
            ArrayD::from_elem(IxDyn(&[2, 3]), 1.),
            None,
            false);
        
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.len(), 6);
        assert_eq!(a.shape(), [2, 3]);
        assert_eq!(a.is_empty(), false);
    }

    #[test]
    fn tensor_addition_test_shape_mismatch() -> Result<(), Box<dyn Error>> {
        let a = Tensor::from_vec(vec![1.; 7*2], vec![7, 2], false)?;    
        let b = Tensor::randn(vec![2, 7], false)?;
        let c = a.add(&b);
        assert!(c.is_err());

        Ok(())
    }

    #[test]
    fn tensor_element_wise_addition_2x2() -> Result<(), Box<dyn Error>> {
        let a = Tensor::from_vec(vec![1., 2., -2., 1.], vec![2, 2], false)?;    
        let b = Tensor::from_vec(vec![2., -5., 1., 6.], vec![2, 2], false)?;
        let c = a.add(&b)?;
        assert_eq!(c.shape(), vec![2, 2]);
        assert_eq!(c.data(), &ArrayD::from_shape_vec(IxDyn(&vec![2, 2]), vec![3., -3., -1., 7.])?);

        let b = Tensor::randn(vec![4, 4], false)?;
        let res = a.add(&b);
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn tensor_element_wise_addition_4x3() -> Result<(), Box<dyn Error>> {
        let a = Tensor::from_vec(vec![1.; 4*3], vec![4, 3], false)?;    
        let b = Tensor::from_vec(vec![2.; 4*3], vec![4, 3], false)?;
        let c = a.add(&b)?;
        assert_eq!(c.shape(), vec![4, 3]);
        assert_eq!(c.data(), &ArrayD::from_shape_vec(IxDyn(&vec![4, 3]), vec![3.; 4*3])?);
        Ok(())
    }

    #[test]
    fn tensor_normal_random() -> Result<(), Box<dyn Error>> {
        let a = Tensor::<f32>::randn(vec![200, 1], false)?;    
        
        let mean_a = a.data().mean();
        assert!(mean_a.is_some());

        let mean_a = mean_a.unwrap();
        assert!(mean_a.abs() < 0.05);

        Ok(())
    }

    #[test]
    fn tensor_uniform_random() -> Result<(), Box<dyn Error>> {
        let a = Tensor::<f32>::rand_uniform(-1., 1., vec![200, 1], false)?;    
        
        let mean_a = a.data().mean();
        assert!(mean_a.is_some());

        let mean_a = mean_a.unwrap();
        assert!(mean_a.abs() < 0.05);

        Ok(())
    }

}