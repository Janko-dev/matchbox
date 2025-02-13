
use std::fmt::Display;

use crate::core::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum Operator<T> {
    // binary
    Add(Tensor<T>, Tensor<T>),
    Sub(Tensor<T>, Tensor<T>),
    Mul(Tensor<T>, Tensor<T>),
    Div(Tensor<T>, Tensor<T>),
    MatMul(Tensor<T>, Tensor<T>),   // default take axis -1 and -2 to perform matmul

    // binary tensor-scalar
    ScalarMul(Tensor<T>, T),
    Pow(Tensor<T>, T),

    // unary with shape/axes arg
    Sum(Tensor<T>, Vec<usize>),
    Mean(Tensor<T>, Vec<usize>),
    Transpose(Tensor<T>, Vec<usize>),
    Broadcast(Tensor<T>, Vec<usize>),

    // proper unary
    Sigmoid(Tensor<T>),
    ReLU(Tensor<T>),
    
}

impl<T> Operator<T> {
    pub fn to_string(&self) -> String {
        match self {
            Self::Add(_, _) => "Add".to_string(),
            Self::Sub(_, _) => "Sub".to_string(),
            Self::Mul(_, _) => "Mul".to_string(),
            Self::Div(_, _) => "Div".to_string(),
            Self::MatMul(_, _) => "MatMul".to_string(),
            
            Self::ScalarMul(_, _) => "Scalar multiplication".to_string(),
            Self::Pow(_, _) => "Scalar exponentiation".to_string(),
            Self::Sum(_, axes) => format!("Sum over axes: {:?}", axes),
            Self::Mean(_, axes) => format!("Mean over axes: {:?}", axes),
            Self::Transpose(_, axes) => format!("Transpose axes: {:?}", axes),
            Self::Broadcast(_, shape) => format!("Broadcast to shape: {:?}", shape),
            Self::Sigmoid(_) => "Sigmoid activation".to_string(),
            Self::ReLU(_) => "ReLU activation".to_string(),
        }
    } 
}

impl<T> Display for Operator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
}