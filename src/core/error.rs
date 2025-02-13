use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatchError {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        op: String,
    },
    BroadCastError {
        got_shape: Vec<usize>,
        expected_shape: Vec<usize>,
    }
}

impl Error for TensorError {}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatchError { a_shape, b_shape, op } => 
                writeln!(f, "Shape mismatch error during [{}] operation: shape {:?} of self does not match shape {:?} of other",
                    op, a_shape, b_shape    
                ),
            TensorError::BroadCastError { got_shape, expected_shape } =>
                writeln!(f, "Broadcast error: could not broadcast shape {:?} into shape {:?}",
                    got_shape, expected_shape    
                )
        }
    }
}