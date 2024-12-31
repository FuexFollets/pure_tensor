use std::marker::PhantomData;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

#[allow(unused_macros)]
macro_rules! ranks_t {
    ($scalar_type:ty, $rank_length:expr, $($other_rank_lengths:expr),*) => {
        [ranks_t!($scalar_type, $($other_rank_lengths),*); $rank_length]
    };

    ($scalar_type:ty, $rank_length:expr) => {
        [$scalar_type; $rank_length]
    };
}

#[allow(unused_macros)]
macro_rules! num_ranks {
    ($scalar_type:ty, $rank_length:expr, $($other_rank_lengths:expr),*) => {
        { 1 + num_ranks!($scalar_type, $($other_rank_lengths),*) }
    };

    ($scalar_type:ty, $rank_length:expr) => {
        { 1 }
    };
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Tensor<ScalarT, RanksT, PermuteInfoT, const N_RANKS: usize> {
    components: RanksT,
    _phantom: PhantomData<(ScalarT, RanksT, PermuteInfoT)>,
}

#[macro_export]
macro_rules! Tensor {
    ($scalar_type:ty, $rank_length:expr, $($other_rank_lengths:expr),*) =>
    {
        Tensor<
            $scalar_type,
            ranks_t!($scalar_type, $rank_length, $($other_rank_lengths),*),
            i32,
            { num_ranks!($scalar_type, $rank_length, $($other_rank_lengths),*) },
        >
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_create_ranks_t() {
        type T = ranks_t!(i32, 10, 3 * 7);
        let x = T::default();
        let len = x[0].len();

        dbg!(len);
    }

    #[test]
    fn test_num_ranks() {
        let number_of_ranks1 = num_ranks!(f64, 44, 3);
        let number_of_ranks2 = num_ranks!(f64, 10, 3 * 7, 290, 1, 1, 44, 3);

        dbg!(number_of_ranks1);
        dbg!(number_of_ranks2);
    }

    /*
    #[test]
    fn test_scalar_t() {
        type T1 = scalar_t!(f64, 1, 2, 3);
        type T2 = scalar_t!(f64, 888, 2, 2, 3);
        type T3 = scalar_t!(std::vec::Vec<f64>, 1, 100202);

        use std::any::type_name;

        dbg!(type_name::<T1>());
        dbg!(type_name::<T2>());
        dbg!(type_name::<T3>());
    }
    */

    #[test]
    fn test_tensor_fields() {
        use super::*;

        #[allow(dead_code)]
        type T = Tensor!(f64, 10, 3, 290, 1, 1, 44, 3);
    }
}
