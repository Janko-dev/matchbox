
use std::{error::Error, rc::Rc, usize};
use rand::{distributions::uniform::SampleUniform, prelude::*};
use crate::core::op::{
    Operator,
        Operator::{
        Add,
        Sub,
        Mul,
        Div,
        MatMul,
        ScalarMul,
        Pow,
        Sum,
        Mean,
        Transpose,
        Broadcast,
        Sigmoid,
        ReLU
    }
};
use crate::core::error::TensorError::{
    BroadCastError,
    ShapeMismatchError, 
    self
};
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::prelude::*;
use ndarray_rand::{
    rand_distr::{num_traits::{Float, One, Zero}, Normal, StandardNormal, Uniform},
    RandomExt
};

// get unique id 
fn get_id() -> usize {
    static COUNTER:AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct Tensor_<T> {
    id: usize,
    data: ArrayD<T>,
    with_grad: bool,
    optype: Option<Operator<T>>
}

#[derive(Debug, Clone)]
pub struct Tensor<T>(Rc<Tensor_<T>>);

// pub type Tensor<T> = Rc<Tensor_<T>>;
pub type TensorResult<T> = Result<Tensor<T>, TensorError>;

impl <T> Tensor_<T> {
    pub fn new(
        data: ArrayD<T>, 
        op: Option<Operator<T>>, 
        with_grad: bool
    ) -> Self {  
        
        Self {
            id: get_id(), 
            data,
            with_grad,
            optype: op 
        }
    }
}

impl<T> Tensor<T> {

    pub fn new(
        data: ArrayD<T>,
        op: Option<Operator<T>>, 
        with_grad: bool
    ) -> Self {

        Self(Rc::new(Tensor_::new(data, op, with_grad)))
    }

    pub fn from_vec(vec_data: Vec<T>, shape: Vec<usize>, with_grad: bool) -> Result<Tensor<T>, Box<dyn Error>> {
        let tensor_data = ArrayD::from_shape_vec(IxDyn(&shape), vec_data)?;
        Ok(Self(Rc::new(Tensor_::new(tensor_data, None, with_grad))))
    }

    pub fn ndim(&self) -> usize {
        self.0.data.ndim()
    }

    pub fn len(&self) -> usize {
        self.0.data.len()
    }
        
    pub fn shape(&self) -> &[usize] {
        self.0.data.shape()
    }
        
    pub fn is_empty(&self) -> bool {
        self.0.data.is_empty()
    }

    pub fn requires_grad(&self) -> bool {
        self.0.with_grad
    }

    pub fn data(&self) -> &ArrayD<T> {
        &self.0.data
    }

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn op(&self) -> &Option<Operator<T>>{
        &self.0.optype
    }

    fn _print_comp_tree(&self, indent: usize) {
        let tensor_info = format!("tensor id: {}, use grad: {}, shape: {:?}", self.id(), self.requires_grad(), self.shape());
        if let Some(op) = self.op() {
            println!("{:indent$}{} with op: {}", " ", tensor_info, op.to_string(), indent=indent);
            match op {
                // binary ops
                Operator::Add(lhs, rhs) |
                Operator::Sub(lhs, rhs) |
                Operator::Mul(lhs, rhs) |
                Operator::Div(lhs, rhs) |
                Operator::MatMul(lhs, rhs)
                => {
                    lhs._print_comp_tree(indent+4);
                    rhs._print_comp_tree(indent+4);
                },
                // binary tensor-scalar + unary with shape/axes arg
                Operator::ScalarMul(val, _) |
                Operator::Pow(val, _) |
                Operator::Sum(val, _) |
                Operator::Mean(val, _) |
                Operator::Transpose(val, _) |
                Operator::Broadcast(val, _)
                => {
                    val._print_comp_tree(indent+4);
                }
                // proper unary
                Operator::Sigmoid(val) |
                Operator::ReLU(val)
                => {
                    val._print_comp_tree(indent+4);
                }
            }
        } else {
            println!("{:indent$}{}", " ", tensor_info, indent=indent);
        }
    }

    pub fn print_comp_tree(&self) {
        
        self._print_comp_tree(0);
    }
}

macro_rules! binary_operator {
    ($name: ident, $op: tt, $op_type: ident) => {
        
        pub fn $name(&self, other: &Self) -> TensorResult<F> {

            if self.shape() != other.shape() {
                return Err(ShapeMismatchError { 
                    a_shape: self.shape().to_vec(), 
                    b_shape: other.shape().to_vec(), 
                    op: stringify!($op_type).to_string() 
                });
            }
    
            let data = &self.0.data $op &other.0.data;
            
            let req_grad = self.requires_grad() || other.requires_grad();
            let op = Some(Operator::$op_type(self.clone(), other.clone()));
    
            Ok(Self::new(data, op, req_grad))
        }
        
    };
}

impl<F: Float + SampleUniform> Tensor<F> {

    pub fn randn(shape: Vec<usize>, with_grad: bool) -> Result<Tensor<F>, Box<dyn Error>> 
    where StandardNormal: Distribution<F>
    {   
        let tensor_data = ArrayD::random(IxDyn(&shape), Normal::<F>::new(F::zero(), F::one())?);
        Ok(Self(Rc::new(Tensor_::new(tensor_data, None, with_grad))))
    }

    pub fn rand_uniform(from: F, to: F, shape: Vec<usize>, with_grad: bool) -> Result<Tensor<F>, Box<dyn Error>> {   
        let tensor_data = ArrayD::random(IxDyn(&shape), Uniform::new(from, to));
        Ok(Self(Rc::new(Tensor_::new(tensor_data, None, with_grad))))
    }

    binary_operator!(add, +, Add);
    binary_operator!(sub, -, Sub);
    binary_operator!(mul, *, Mul);
    binary_operator!(div, /, Div);
    
    // pub fn matmul(&self, other: &Self) -> TensorResult<F> {

    //     // TODO: implement matmul, check for dimensions and default perform on axis -1 and -2

    //     // if self.shape() != other.shape() {
    //     //     return Err(ShapeMismatchError { 
    //     //         a_shape: self.shape().to_vec(), 
    //     //         b_shape: other.shape().to_vec(), 
    //     //         op: "Matrix multiply".to_string() 
    //     //     });
    //     // }

    //     // let data = &self.0.data $op &other.0.data;
        
    //     // let req_grad = self.requires_grad() || other.requires_grad();
    //     // let op = Some(Operator::$op_type(self.clone(), other.clone()));

    //     // Ok(Self::new(data, op, req_grad))
    //     // Err()
    // }
}