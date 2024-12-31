use std::marker::PhantomData;
// use std::ops::{Add, Div, Index, Mul, Neg, Sub};

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
macro_rules! permutation {
    ($first_new_rank:expr, $($new_rank_for_each_index:expr),*) => {
        [permutation!($($new_rank_for_each_index),*); { $first_new_rank }]
    };

    ($new_rank_for_each_index:expr) => {
        [(); { $new_rank_for_each_index }]
    };
}

// cursed
#[allow(unused_macros)]
macro_rules! default_permutation {
    ($num_ranks:expr, $separator_1:ty,
     $subtractor:expr, $separator_2:ty,
     $rank_length:expr, $($other_rank_lengths:expr),*) => {
        [
            default_permutation!(
                $num_ranks,
                (),
                ($subtractor - 1),
                (),
                $($other_rank_lengths),*
                );
            { $num_ranks - $subtractor }
        ]
    };

    ($num_ranks:expr, $separator_1:ty,
     $subtractor:expr, $separator_2:ty,
     $rank_length:expr) => {
        [(); { $num_ranks - $subtractor }]
    };

    ($scalar_type:ty, $rank_length:expr, $($other_rank_lengths:expr),*) => {
        default_permutation!(
            num_ranks!($scalar_type, $rank_length, $($other_rank_lengths),*),
            (),
            num_ranks!($scalar_type, $rank_length, $($other_rank_lengths),*),
            (),
            $rank_length,
            $($other_rank_lengths),*
        )
    };
}

#[macro_export]
macro_rules! tensor {
    ($scalar_type:ty, $rank_length:expr, $($other_rank_lengths:expr),*) =>
    {
        Tensor<
            $scalar_type,
            ranks_t!($scalar_type, $rank_length, $($other_rank_lengths),*),
            default_permutation!($scalar_type, $rank_length, $($other_rank_lengths),*),
            { num_ranks!($scalar_type, $rank_length, $($other_rank_lengths),*) },
        >
    };

    ($scalar_type:ty, $permutation:ty, $rank_length:expr, $($other_rank_lengths:expr),*) =>
    {
        Tensor<
            $scalar_type,
            ranks_t!($scalar_type, $rank_length, $($other_rank_lengths),*),
            $permutation,
            { num_ranks!($scalar_type, $rank_length, $($other_rank_lengths),*) },
        >
    };
}

pub trait NumRanks {
    const NUM_RANKS: usize;

    fn num_ranks() -> usize {
        Self::NUM_RANKS
    }
}

impl<T, const N: usize> NumRanks for [T; N]
where
    T: NumRanks,
{
    const NUM_RANKS: usize = 1 + T::NUM_RANKS;
}

impl NumRanks for () {
    const NUM_RANKS: usize = 0;
}

impl<ScalarT, RanksT, PermuteInfoT, const N_RANKS: usize> NumRanks
    for Tensor<ScalarT, RanksT, PermuteInfoT, N_RANKS>
where
    PermuteInfoT: NumRanks,
{
    const NUM_RANKS: usize = PermuteInfoT::NUM_RANKS;
}

impl<ScalarT, RanksT, PermuteInfoT, const N_RANKS: usize> Default
    for Tensor<ScalarT, RanksT, PermuteInfoT, N_RANKS>
where
    RanksT: Default,
{
    fn default() -> Self {
        Self {
            components: RanksT::default(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        #[allow(dead_code)]
        type T = tensor!(f64, 10, 3, 290, 1, 1, 44, 3);
        use std::any::type_name;

        dbg!(type_name::<T>());
    }

    #[test]
    fn test_default_permutation() {
        type P = default_permutation!(f64, 10, 3, 290, 1, 1, 44, 3);
        use std::any::type_name;

        dbg!(type_name::<P>());
        dbg!(P::NUM_RANKS);
    }

    #[test]
    fn test_tensor_meta() {
        type T = tensor!(f64, 10, 3, 290, 1, 1, 44, 3);
        use std::any::type_name;

        dbg!(type_name::<T>());
        dbg!(T::NUM_RANKS);
    }
}
