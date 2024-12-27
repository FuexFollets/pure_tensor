use std::ops::{Add, Div, Index, Mul, Neg, Sub};

/* Notes:
 * Each of the components of the tensor transforms the basis of the operand space
 */

// pub trait Tensor<TComponents, const N: usize>: Index<usize, Output = TComponents> {}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Tensor<TComponents, const N: usize> {
    components: [TComponents; N],
}

impl<TComponents, const N: usize> Tensor<TComponents, N> {
    pub fn new(components: [TComponents; N]) -> Self {
        Tensor { components }
    }
}

impl<TComponents, const N: usize> Index<usize> for Tensor<TComponents, N> {
    type Output = TComponents;
    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<TComponents, const N: usize> Default for Tensor<TComponents, N>
where
    TComponents: Default + Copy,
{
    fn default() -> Self {
        Tensor {
            components: [TComponents::default(); N],
        }
    }
}

impl<TComponentsLhs, TComponentsRhs, const N: usize> Add<Tensor<TComponentsRhs, N>>
    for Tensor<TComponentsLhs, N>
where
    TComponentsLhs: Add<TComponentsRhs> + Copy,
    TComponentsRhs: Copy,
    <TComponentsLhs as Add<TComponentsRhs>>::Output: Default + Copy,
{
    type Output = Tensor<<TComponentsLhs as Add<TComponentsRhs>>::Output, N>;

    fn add(self, rhs: Tensor<TComponentsRhs, N>) -> Self::Output {
        let mut sum = Self::Output::default();

        for i in 0..N {
            sum.components[i] = self.components[i] + rhs.components[i];
        }

        sum
    }
}

impl<TComponentsLhs, TComponentsRhs, const N: usize> Sub<Tensor<TComponentsRhs, N>>
    for Tensor<TComponentsLhs, N>
where
    TComponentsLhs: Sub<TComponentsRhs> + Copy,
    TComponentsRhs: Copy,
    <TComponentsLhs as Sub<TComponentsRhs>>::Output: Default + Copy,
{
    type Output = Tensor<<TComponentsLhs as Sub<TComponentsRhs>>::Output, N>;

    fn sub(self, rhs: Tensor<TComponentsRhs, N>) -> Self::Output {
        let mut sum = Self::Output::default();

        for i in 0..N {
            sum.components[i] = self.components[i] - rhs.components[i];
        }

        sum
    }
}

impl<TComponents, const N: usize, TScalar> Mul<TScalar> for Tensor<TComponents, N>
where
    TComponents: Mul<TScalar> + Copy,
    TScalar: Copy,
    <TComponents as Mul<TScalar>>::Output: Default + Copy,
{
    type Output = Tensor<<TComponents as Mul<TScalar>>::Output, N>;

    fn mul(self, rhs: TScalar) -> Self::Output {
        let mut product = Self::Output::default();

        for i in 0..N {
            product.components[i] = self.components[i] * rhs;
        }

        product
    }
}

impl<TComponents, const N: usize, TScalar> Div<TScalar> for Tensor<TComponents, N>
where
    TComponents: Div<TScalar> + Copy,
    TScalar: Copy,
    <TComponents as Div<TScalar>>::Output: Default + Copy,
{
    type Output = Tensor<<TComponents as Div<TScalar>>::Output, N>;

    fn div(self, rhs: TScalar) -> Self::Output {
        let mut quotient = Self::Output::default();

        for i in 0..N {
            quotient.components[i] = self.components[i] / rhs;
        }

        quotient
    }
}

impl<TComponents, const N: usize> Neg for Tensor<TComponents, N>
where
    TComponents: Neg + Copy,
    <TComponents as Neg>::Output: Default + Copy,
{
    type Output = Tensor<<TComponents as Neg>::Output, N>;

    fn neg(self) -> Self::Output {
        let mut negated = Self::Output::default();

        for i in 0..N {
            negated.components[i] = -self.components[i];
        }

        negated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::f64::{Length, Time};
    use uom::si::{length, time};

    #[test]
    fn test_tensor() {
        let tensor = Tensor::new([1, 2, 3]);

        let tensor2 = Tensor::new([4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_basic_arithmetic_uom() {
        let one_meter = Length::new::<length::meter>(1.0);
        let two_inches = Length::new::<length::inch>(2.0);
        let three_feet = Length::new::<length::foot>(3.0);

        let one_second = Time::new::<time::second>(1.0);
        let two_minutes = Time::new::<time::minute>(2.0);
        let three_hours = Time::new::<time::hour>(3.0);

        let tensor = Tensor::new([one_meter, two_inches, three_feet]);
        let tensor2 = tensor.clone() * 2.0;
        let tensor3 = Tensor::new([one_second, two_minutes, three_hours]);

        let sum = tensor + tensor2;
        let difference = tensor - tensor2;
        let product = tensor * 2.0;
        let quotient = tensor / 2.0;

        println!("Tensor: {:?}", tensor);
        println!("Tensor2: {:?}", tensor2);
        println!("Tensor3: {:?}", tensor3);
        println!("Sum: {:?}", sum);
        println!("Difference: {:?}", difference);
        println!("Product: {:?}", product);
        println!("Quotient: {:?}", quotient);
    }

    #[test]
    fn test_ndarray_uom() {
        let one_meter = Length::new::<length::meter>(1.0);
        let two_inches = Length::new::<length::inch>(2.0);
        let three_feet = Length::new::<length::foot>(3.0);

        let one_second = Time::new::<time::second>(1.0);
        let two_minutes = Time::new::<time::minute>(2.0);
        let three_hours = Time::new::<time::hour>(3.0);

        let arr_lengths = ndarray::arr1(&[one_meter, two_inches, three_feet]);
        let arr_times = ndarray::arr1(&[one_second, two_minutes, three_hours]);

        println!("Arr_lengths: {:?}", arr_lengths);
        println!("Arr_times: {:?}", arr_times);

        // let sum = arr_lengths.clone() + arr_lengths.clone();
        // let difference = arr_lengths - arr_lengths;

        // println!("Sum: {:?}", sum);
        // println!("Difference: {:?}", difference);

        // let dot = move || arr_lengths.clone().dot(&arr_times);

        println!("Dot: {:?}", dot);
    }
}
