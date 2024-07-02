//! # miniradixsort
//!
//! Some simple unstable byte-based counting / radix sort functions.
//!
//! Ported from
//!
//! [`https://probablydance.com/2016/12/27/i-wrote-a-faster-sorting-algorithm/`]()
//! [`https://github.com/skarupke/ska_sort`]()
//!
//! Initial motivation: in-place byte slice suffix array sorting for string matching purposes.

use {
    miniunchecked::*,
    miniunsigned::{NonZero, Unsigned},
    std::cmp::Ordering,
};

/// Number of unique byte values
/// and thus unique buckets in any slice sorted based on a byte key.
pub const NUM_BUCKETS: usize = u8::MAX as usize + 1;

/// Unstable non-in-place (i.e. copying) (byte) bucket / counting sort; a.k.a. non-in-place (i.e. copying) byte radix sort by the first byte / radix "digit".
///
/// (Partially, for larger-than-byte types) sorts the `slice` by cloning / copying the elements into the `result` slice in the (unstable) order
/// given by comparing the bytes extracted from the slice elements by the provided `b` closure.
///
/// If `result` is longer than the maximum value of the size type `S`, only the first `S::max_value()` elements will be processed.
///
/// `slice` is a closure which is given an element index in range `[0 .. result.len())` (or `[0 .. S::max_value()]`, whichever is smaller)
/// and returns a (potentially cloned) slice element at that index.
///
/// `b` is a closure which is given a slice element and extracts / returns a byte for this element
/// for byte bucket counting purposes.
///
/// Returns an array of byte bucket end indices in the `result` array (i.e. indices past the last element in the bucket).
///
/// May be used to initialize the partially sorted `result` slice for radix sort to operate upon (by [`radix_sort_lvl`], with `lvl = 1`)
/// in cases where it is more efficient than constructing the unsorted slice and then radix-sorting it.
///
/// Example - suffix array construction. In this case `T == S` (`slice` elements are suffix indices),
/// `slice` is a passthrough closure which returns its argument - the suffix index,
/// and `b` returns the first byte of the suffix at a given index from the source text the suffix array is constructed for.
/// This way `result` is initialized directly to the partially-sorted (by the first byte) suffix array of the source text,
/// instead of having to first construct an unsorted suffix array (i.e. a sequence `[0, 1, 2, .., N]`) and then radix-sorting it
/// starting with `lvl = 0`.
pub fn counting_sort<S, F, T, B>(slice: F, result: &mut [T], b: &B) -> [S; NUM_BUCKETS]
where
    S: Unsigned,
    F: Fn(S) -> T,
    B: Fn(&T) -> u8,
{
    let result_len = S::from_usize(result.len()).unwrap_or(S::max_value());
    // Safe because `result_len <= result.len()`.
    let result = unsafe { result.get_unchecked_mut(..result_len.to_usize()) };

    let mut bucket_sizes = [S::zero(); NUM_BUCKETS];

    for i in 0..result_len.to_usize() {
        // Safe because `i <= result_len <= S::max_value()`.
        bucket_sizes[b(&slice(unsafe { S::from_usize(i).unwrap_unchecked_dbg() })) as usize] +=
            S::one();
    }

    let mut bucket_starts = bucket_sizes;
    {
        let mut sum = S::zero();
        for b_start in bucket_starts.iter_mut() {
            let b_size = *b_start;
            *b_start = sum;
            sum += b_size;
        }
    }

    for i in 0..result_len.to_usize() {
        // Safe because `result_len <= S::max_value()`.
        let element = slice(unsafe { S::from_usize(i).unwrap_unchecked_dbg() });
        let pos = &mut bucket_starts[b(&element) as usize];
        result[pos.to_usize()] = element;
        *pos += S::one();
    }

    let bucket_ends = bucket_starts;
    bucket_ends
}

/// See [`counting_sort`].
///
/// This is a convenience wrapper to sort a `slice` directly if it is available.
/// `slice` and `result` should have the same length, otherwise only the number of elements
/// equal to the lowest length among them will be sorted.
pub fn counting_sort_slice<S, T, B>(slice: &[T], result: &mut [T], b: &B) -> [S; NUM_BUCKETS]
where
    S: Unsigned,
    T: Clone,
    B: Fn(&T) -> u8,
{
    let len = slice.len().min(result.len());
    counting_sort::<S, _, _, _>(|i: S| slice[i.to_usize()].clone(), &mut result[..len], b)
}

/// Use the standard library (unstable, quick)sort for slices shorter than this threshold.
const STD_SORT_THRESHOLD: usize = 128;

/// Use the "american flag" byte radix sort implementation for slices shorter than this threshold.
/// Use the "ska" byte radix sort for longer slices.
const AMERICAN_FLAG_SORT_THRESHOLD: usize = 1024;

/// Unstable byte radix sort.
///
/// Radix sorts the slice by bytes / radix "digits" in range `[0 .. num_lvls)`.
///
/// `b` is a predicate which extracts the byte at given radix / byte "digit" index from the `slice` element,
/// from most to least significant.
/// It is passed the slice element to extract the byte from and the radix index.
///
/// `cmp` is a predicate which compares the given slice elements.
///
/// `num_lvls` is the number of bytes / radix "digits" to sort by.
///
/// E.g., when sorting bytes (`T` is `u8`), `num_lvls` is `1` and `lvl` is always `0`.
/// When sorting words (`T` is `u16`), `num_lvls` is `2` and `lvl` may be `0` for high byte and `1` for low byte.
/// When sorting dwords (`T` is `u32`), `num_lvls` is `4` and `lvl` may be in range `[0 .. 3]`, for bytes from highest to lowest; etc.
///
/// Ported from
///
/// [`https://probablydance.com/2016/12/27/i-wrote-a-faster-sorting-algorithm/`]()
/// [`https://github.com/skarupke/ska_sort`]()
pub fn radix_sort<S, N, T, B, C>(slice: &mut [T], b: &B, cmp: &C, num_lvls: N)
where
    S: Unsigned,
    N: NonZero<S>,
    B: Fn(&T, S) -> u8,
    C: Fn(&T, &T, S) -> Ordering,
{
    radix_sort_lvl(slice, b, cmp, S::zero(), num_lvls);
}

/// See [`radix_sort`].
///
/// Radix sorts the slice by bytes / radix "digits" in range `[lvl .. num_lvls)`.
/// Does nothing if `lvl >= num_lvls`.
pub fn radix_sort_lvl<S, N, T, B, C>(slice: &mut [T], b: &B, cmp: &C, lvl: S, num_lvls: N)
where
    S: Unsigned,
    N: NonZero<S>,
    B: Fn(&T, S) -> u8,
    C: Fn(&T, &T, S) -> Ordering,
{
    if lvl >= num_lvls.get() {
        return;
    }

    radix_sort_impl::<STD_SORT_THRESHOLD, AMERICAN_FLAG_SORT_THRESHOLD, S, N, _, _, _>(
        slice, b, cmp, lvl, num_lvls,
    );
}

fn radix_sort_impl<const ST: usize, const AT: usize, S, N, T, B, C>(
    slice: &mut [T],
    b: &B,
    cmp: &C,
    lvl: S,
    num_lvls: N,
) where
    S: Unsigned,
    N: NonZero<S>,
    B: Fn(&T, S) -> u8 + ?Sized,
    C: Fn(&T, &T, S) -> Ordering,
{
    debug_assert!(lvl < num_lvls.get());

    if slice.len() < ST {
        std_sort(slice, cmp, lvl, num_lvls);
    } else if slice.len() < AT {
        american_flag_sort::<ST, AT, S, N, _, _, _>(slice, b, cmp, lvl, num_lvls);
    } else {
        ska_sort::<ST, AT, S, N, _, _, _>(slice, b, cmp, lvl, num_lvls);
    }
}

fn std_sort<S, N, T, C>(slice: &mut [T], cmp: &C, lvl: S, num_lvls: N)
where
    S: Unsigned,
    N: NonZero<S>,
    C: Fn(&T, &T, S) -> Ordering,
{
    debug_assert!(lvl < num_lvls.get());

    slice.sort_unstable_by(|l, r| cmp(l, r, lvl))
}

fn bucket_sizes<S, T, B>(slice: &mut [T], b: &B, lvl: S) -> [S; NUM_BUCKETS]
where
    S: Unsigned,
    B: Fn(&T, S) -> u8 + ?Sized,
{
    let mut bucket_sizes = [S::zero(); NUM_BUCKETS];

    for element in slice.iter() {
        bucket_sizes[b(element, lvl) as usize] += S::one();
    }

    bucket_sizes
}

fn american_flag_sort<const ST: usize, const AT: usize, S, N, T, B, C>(
    slice: &mut [T],
    b: &B,
    cmp: &C,
    lvl: S,
    num_lvls: N,
) where
    S: Unsigned,
    N: NonZero<S>,
    B: Fn(&T, S) -> u8 + ?Sized,
    C: Fn(&T, &T, S) -> Ordering,
{
    debug_assert!(lvl < num_lvls.get());

    let slice_len = S::from_usize(slice.len()).unwrap_or(S::max_value());
    let slice = unsafe { slice.get_unchecked_mut(..slice_len.to_usize()) };

    // Calculate the byte bucket sizes at the radix `lvl`.
    let bucket_sizes = bucket_sizes(slice, b, lvl);

    // Using the `bucket_sizes`, calculate the start/end indices (`bucket_starts` / `bucket_ends`) for byte buckets at radix `lvl`,
    // as well as the indices (`remaining_buckets`) and the total number (`num_remaining_buckets`) of existing buckets.
    let mut bucket_starts = bucket_sizes;
    let mut bucket_ends = [S::zero(); NUM_BUCKETS];
    let mut remaining_buckets = [0u8; NUM_BUCKETS];
    let mut num_remaining_buckets = S::zero();

    {
        let mut sum = S::zero();
        for (b_idx, (b_start, b_end)) in bucket_starts
            .iter_mut()
            .zip(bucket_ends.iter_mut())
            .enumerate()
        {
            let b_size = *b_start;
            // Skip empty buckets. These will have `0` entries in `bucket_starts` / `bucket_ends`.
            if b_size > S::zero() {
                *b_start = sum;
                sum += b_size;

                remaining_buckets[num_remaining_buckets.to_usize()] = b_idx as u8;
                num_remaining_buckets += S::one();

                *b_end = sum;
            }
        }
    }

    // No need to do anything if we only have one bucket - it's already sorted.
    if num_remaining_buckets > S::one() {
        let mut remaining_bucket_index = S::zero();
        let mut remaining_bucket_end =
            bucket_ends[remaining_buckets[remaining_bucket_index.to_usize()] as usize];
        let mut index = S::zero();

        'recursion_level: loop {
            let byte = b(&slice[index.to_usize()], lvl);
            let bucket_start = bucket_starts[byte as usize];
            let bucket_end = bucket_ends[byte as usize];

            // If the current slice element is in correct bucket, go to the next one.
            if (bucket_start..bucket_end).contains(&index) {
                // For sanity check purposes below.
                #[cfg(debug_assertions)]
                {
                    bucket_starts[byte as usize] += S::one();
                }

                index += S::one();

                // We've sorted all slice elements - we're done with this recursion level.
                if index == slice_len {
                    break 'recursion_level;

                // We've processed all slice elements in the current remaining bucket - find the next unsorted remaining bucket.
                } else if index == remaining_bucket_end {
                    'find_next_bucket: loop {
                        remaining_bucket_index += S::one();

                        // We've processed all remaining buckets - we're done with this recursion level.
                        if remaining_bucket_index == num_remaining_buckets {
                            break 'recursion_level;
                        }

                        let remaining_bucket =
                            remaining_buckets[remaining_bucket_index.to_usize()] as usize;

                        let remaining_bucket_start = bucket_starts[remaining_bucket];
                        remaining_bucket_end = bucket_ends[remaining_bucket];

                        // We found the next unsorted remaining bucket.
                        if remaining_bucket_start != remaining_bucket_end {
                            index = remaining_bucket_start;
                            debug_assert!(index < slice_len);
                            break 'find_next_bucket;
                        }
                    }
                }

            // If the current slice element is in a wrong bucket, swap it with an element in that bucket.
            // Process that element on the next iteration.
            } else {
                slice.swap(index.to_usize(), bucket_start.to_usize());
                bucket_starts[byte as usize] += S::one();
                debug_assert!(bucket_starts[byte as usize] <= bucket_ends[byte as usize]);
            }
        }
    } else if num_remaining_buckets == S::one() {
        // For sanity check purposes below.
        #[cfg(debug_assertions)]
        {
            let bi = remaining_buckets[0] as usize;
            bucket_starts[bi] = bucket_ends[bi];
        }
    }

    // Sanity check.
    #[cfg(debug_assertions)]
    {
        for rbi in 0..num_remaining_buckets.to_usize() {
            let bi = remaining_buckets[rbi] as usize;
            debug_assert!(bucket_starts[bi] == bucket_ends[bi]);
        }
    }

    // Recursively sort all the remaining buckets by the next byte / radix level.
    let next_lvl = lvl + S::one();
    if next_lvl < num_lvls.get() {
        let mut bucket_start = S::zero();
        for &bi in remaining_buckets
            .iter()
            .take(num_remaining_buckets.to_usize())
        {
            let bucket_end = bucket_ends[bi as usize];
            debug_assert!(bucket_end > bucket_start);
            let bucket_size = bucket_end - bucket_start;
            if bucket_size > S::one() {
                radix_sort_impl::<ST, AT, S, N, _, _, _>(
                    &mut slice[bucket_start.to_usize()..bucket_end.to_usize()],
                    b,
                    cmp,
                    next_lvl,
                    num_lvls,
                );
            }
            bucket_start = bucket_end;
        }
    }
}

fn ska_sort<const ST: usize, const AT: usize, S, N, T, B, C>(
    slice: &mut [T],
    b: &B,
    cmp: &C,
    lvl: S,
    num_lvls: N,
) where
    S: Unsigned,
    N: NonZero<S>,
    B: Fn(&T, S) -> u8 + ?Sized,
    C: Fn(&T, &T, S) -> Ordering,
{
    debug_assert!(lvl < num_lvls.get());

    let slice_len = S::from_usize(slice.len()).unwrap_or(S::max_value());
    let slice = unsafe { slice.get_unchecked_mut(..slice_len.to_usize()) };

    // Calculate the byte bucket sizes at the radix `lvl`.
    let bucket_sizes = bucket_sizes(slice, b, lvl);

    // Using the `bucket_sizes`, calculate the start/end indices (`bucket_starts` / `bucket_ends`) for byte buckets at radix `lvl`,
    // as well as the indices (`remaining_buckets`) and the total number (`num_remaining_buckets`) of existing buckets.
    let mut bucket_starts = bucket_sizes;
    let mut bucket_ends = [S::zero(); NUM_BUCKETS];
    let mut remaining_buckets = [0u8; NUM_BUCKETS];
    let mut num_remaining_buckets = S::zero();

    {
        let mut sum = S::zero();
        for (b_idx, (b_start, b_end)) in bucket_starts
            .iter_mut()
            .zip(bucket_ends.iter_mut())
            .enumerate()
        {
            let b_size = *b_start;
            if b_size > S::zero() {
                *b_start = sum;
                sum += b_size;

                remaining_buckets[num_remaining_buckets.to_usize()] = b_idx as u8;
                num_remaining_buckets += S::one();
            }
            *b_end = sum;
        }
    }

    let mut num_remaining_unsorted_buckets = num_remaining_buckets;
    loop {
        // Partition the remaining buckets into unsorted (at the start, we'll process those) and sorted (at the end, we don't care about these).
        // Get the index past the last unsorted (or first sorted) remaining bucket (which is also equal to the number of remaining unsorted buckets).
        num_remaining_unsorted_buckets = slice_partition_index(
            &mut remaining_buckets[0..num_remaining_unsorted_buckets.to_usize()],
            |&b_idx| {
                let bucket_start = bucket_starts[b_idx as usize];
                let bucket_end = bucket_ends[b_idx as usize];

                // The bucket is sorted, put it at the end partition.
                let is_sorted = bucket_start == bucket_end;
                if is_sorted {
                    return false;
                }

                // The bucket is unsorted (yet) - do up to four iterations of unconditional swaps.
                unroll_4(bucket_start, bucket_end - bucket_start, |index| {
                    let byte = b(&slice[index.to_usize()], lvl);
                    let offset = &mut bucket_starts[byte as usize];
                    slice.swap(index.to_usize(), offset.to_usize());
                    *offset += S::one();
                });

                // Put the bucket at the end partition if it is now sorted.
                let is_sorted = bucket_starts[b_idx as usize] == bucket_ends[b_idx as usize];
                return !is_sorted;
            },
        );

        // If there's one "unsorted" bucket left, it means it's sorted (because all other buckets are).
        // All remaining buckets are sorted - we're done with this byte / radix level.
        if num_remaining_unsorted_buckets <= S::one() {
            // For sanity check below.
            #[cfg(debug_assertions)]
            {
                if num_remaining_unsorted_buckets == S::one() {
                    let bi = remaining_buckets[0] as usize;
                    bucket_starts[bi] = bucket_ends[bi];
                }
            }
            break;
        }
    }

    // Sanity check.
    #[cfg(debug_assertions)]
    {
        for rbi in 0..num_remaining_buckets.to_usize() {
            let bi = remaining_buckets[rbi] as usize;
            debug_assert!(bucket_starts[bi] == bucket_ends[bi]);
        }
    }

    // Recursively sort all the remaining buckets by the next byte / radix level.
    let next_lvl = lvl + S::one();
    if next_lvl < num_lvls.get() {
        let bucket_start = |b: u8| {
            if b == 0 {
                S::zero()
            } else {
                bucket_ends[(b - 1) as usize]
            }
        };
        let bucket_end = |b: u8| bucket_ends[b as usize];

        for &bi in remaining_buckets
            .iter()
            .take(num_remaining_buckets.to_usize())
        {
            let bucket_start = bucket_start(bi);
            let bucket_end = bucket_end(bi);
            debug_assert!(bucket_end > bucket_start);
            let bucket_size = bucket_end - bucket_start;
            if bucket_size > S::one() {
                radix_sort_impl::<ST, AT, S, N, _, _, _>(
                    &mut slice[bucket_start.to_usize()..bucket_end.to_usize()],
                    b,
                    cmp,
                    next_lvl,
                    num_lvls,
                );
            }
        }
    }
}

/// Partitions (unstably) the `slice` in-place into two sections,
/// first all elements for which the predicate `f` returns `true`,
/// followed by all elements for which the predicate `f` returns `false`.
///
/// Returns the index of the first element in the second section / just past the last element in the first section.
fn slice_partition_index<S, T, F>(slice: &mut [T], mut f: F) -> S
where
    S: Unsigned,
    F: FnMut(&T) -> bool,
{
    let slice_len = S::from_usize(slice.len()).unwrap_or(S::max_value());

    // let len = slice.len();
    // if len == 0 {
    //     return 0;
    // }
    // let mut l = 0;
    // let mut r = len - 1;
    // loop {
    //     while l < len && f(&slice[l]) {
    //         l += 1;
    //     }
    //     while r > 0 && !f(&slice[r]) {
    //         r -= 1;
    //     }
    //     if l >= r {
    //         return l;
    //     }
    //     slice.swap(l, r);
    // }

    let mut l = S::zero();

    loop {
        if l == slice_len {
            return l;
        }

        if !f(&slice[l.to_usize()]) {
            break;
        }

        l += S::one();
    }

    let mut r = l + S::one();

    loop {
        if r == slice_len {
            return l;
        }

        if f(&slice[r.to_usize()]) {
            slice.swap(l.to_usize(), r.to_usize());
            l += S::one();
        }

        r += S::one();
    }
}

fn unroll_4<S, F>(mut index: S, num: S, mut f: F)
where
    S: Unsigned,
    F: FnMut(S),
{
    let four = unsafe { S::from_usize(4).unwrap_unchecked_dbg() };
    let num_loops = num / four;
    for _ in 0..num_loops.to_usize() {
        f(index);
        index += S::one();
        f(index);
        index += S::one();
        f(index);
        index += S::one();
        f(index);
        index += S::one();
    }
    let mut remainder = num - num_loops * four;
    while remainder > S::zero() {
        f(index);
        index += S::one();
        remainder -= S::one();
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        rand::{distributions::Distribution, seq::SliceRandom, Rng, SeedableRng},
        std::num::{NonZeroU16, NonZeroU32, NonZeroU8},
    };

    #[test]
    fn counting_sort_test() {
        const LEN: usize = 8;

        let unsorted: [u8; LEN] = [3, 0, 1, 7, 5, 4, 2, 6];
        let sorted: [u8; LEN] = [0, 1, 2, 3, 4, 5, 6, 7];

        // [1, 2, 3, 4, 5, 6, 7, 8, 8, ..., 8]
        let mut expected_bucket_ends = [0u8; NUM_BUCKETS];
        for i in 0..NUM_BUCKETS {
            expected_bucket_ends[i] = (i + 1).min(8) as _;
        }

        {
            let mut result = [0; LEN];

            let bucket_ends = counting_sort::<u8, _, _, _>(
                |i: u8| unsorted[i as usize],
                &mut result,
                &|val: &u8| *val,
            );

            assert_eq!(result, sorted);
            assert_eq!(bucket_ends, expected_bucket_ends);
        }

        {
            let mut result = [0; LEN];

            let bucket_ends =
                counting_sort_slice::<u8, _, _>(&unsorted, &mut result, &|val: &u8| *val);

            assert_eq!(result, sorted);
            assert_eq!(bucket_ends, expected_bucket_ends);
        }
    }

    #[test]
    fn counting_sort_test_u8() {
        const LEN: usize = u8::MAX as usize;

        // [0, 1, 2, .., 254]
        let sorted = {
            let mut sorted = [0u8; LEN];
            for (i, value) in sorted.iter_mut().enumerate() {
                *value = i as u8;
            }
            sorted
        };

        // [1, 2, .., 255, 255]
        let mut expected_bucket_ends = [0u8; NUM_BUCKETS];
        for i in 0..NUM_BUCKETS {
            expected_bucket_ends[i] = (i + 1).min(u8::MAX as usize) as _;
        }

        let num_iterations = 1000;

        for i in 0..num_iterations {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i);

            let unsorted = {
                let mut unsorted = sorted;
                unsorted.shuffle(&mut rng);
                unsorted
            };

            let mut result = [0; LEN];
            let bucket_ends =
                counting_sort_slice::<u8, _, _>(&unsorted, &mut result, &|val: &u8| *val as _);

            assert_eq!(result, sorted);
            assert_eq!(bucket_ends, expected_bucket_ends);
        }
    }

    #[test]
    fn slice_partition_index_test() {
        let mut slice = [0, 1, 2, 3, 4, 5, 6, 7];
        // Partition into evens and odds.
        let is_even = |val: &u8| val % 2 == 0;
        let idx = slice_partition_index(&mut slice, is_even);
        assert_eq!(idx, 4);
        for i in 0..slice.len() {
            assert!(is_even(&slice[i]) == (i < idx));
        }
        // Now do the reverse.
        let is_odd = |val: &u8| !is_even(val);
        let idx = slice_partition_index(&mut slice, is_odd);
        assert_eq!(idx, 4);
        for i in 0..slice.len() {
            assert!(is_odd(&slice[i]) == (i < idx));
        }
    }

    fn sort_unique_bytes<F>(mut f: F)
    where
        F: FnMut(&mut [u8]),
    {
        let mut unsorted = [3, 0, 1, 7, 5, 4, 2, 6];
        let sorted = [0, 1, 2, 3, 4, 5, 6, 7];

        f(&mut unsorted);

        assert_eq!(unsorted, sorted);
    }

    fn sort_unique_bytes_u8<F>(mut f: F)
    where
        F: FnMut(&mut [u8]),
    {
        // [0, 1, 2, .., 254]
        let sorted = {
            let mut sorted = [0u8; u8::MAX as usize];
            for (i, value) in sorted.iter_mut().enumerate() {
                *value = i as u8;
            }
            sorted
        };

        let num_iterations = 100;

        for i in 0..num_iterations {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i);

            let mut unsorted = {
                let mut unsorted = sorted;
                unsorted.shuffle(&mut rng);
                unsorted
            };

            f(&mut unsorted);

            assert_eq!(unsorted, sorted);
        }
    }

    fn generate_random_bytes<R: Rng>(rng: &mut R) -> Vec<u8> {
        use rand::distributions::Uniform;

        let size = match Uniform::new_inclusive(0u8, 2).sample(rng) {
            0 => Uniform::new_inclusive(128u8, u8::MAX).sample(rng) as usize,
            1 => Uniform::new_inclusive(u8::MAX as u16 + 1, u16::MAX).sample(rng) as usize,
            2 => Uniform::new_inclusive(u16::MAX as u32 + 1, (u16::MAX as u32) * 2).sample(rng)
                as usize,
            _ => unreachable!(),
        };

        let random_byte_distr = Uniform::new_inclusive(0u8, u8::MAX);

        (0..size).map(|_| random_byte_distr.sample(rng)).collect()
    }

    fn sort_random_bytes<FU8, FU16, FU32>(mut fu8: FU8, mut fu16: FU16, mut fu32: FU32)
    where
        FU8: FnMut(&mut [u8]),
        FU16: FnMut(&mut [u8]),
        FU32: FnMut(&mut [u8]),
    {
        let num_iterations = 100;

        for i in 0..num_iterations {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i);

            let mut unsorted = generate_random_bytes(&mut rng);
            let sorted = {
                let mut sorted = unsorted.clone();
                sorted.sort_unstable();
                sorted
            };

            if unsorted.len() <= u8::MAX as usize {
                fu8(&mut unsorted);
            } else if unsorted.len() <= u16::MAX as usize {
                fu16(&mut unsorted);
            } else {
                fu32(&mut unsorted);
            }

            assert_eq!(unsorted, sorted);
        }
    }

    fn sort_unique_tuples<F>(mut f: F)
    where
        F: FnMut(&mut [(u8, u8)]),
    {
        let mut unsorted = [
            (3, 4),
            (0, 1),
            (1, 2),
            (7, 8),
            (5, 6),
            (4, 5),
            (2, 3),
            (6, 7),
        ];
        let sorted = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
        ];

        f(&mut unsorted);

        assert_eq!(unsorted, sorted);
    }

    fn sort_unique_tuples_u8<F>(mut f: F)
    where
        F: FnMut(&mut [(u8, u8)]),
    {
        // [(0, 1), (1, 2), (2, 3) .., (254, 255)]
        let sorted = {
            let mut sorted = [(0u8, 0u8); u8::MAX as usize];
            for (i, value) in sorted.iter_mut().enumerate() {
                *value = (i as u8, i as u8 + 1);
            }
            sorted
        };

        let num_iterations = 100;

        for i in 0..num_iterations {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i);

            let mut unsorted = {
                let mut unsorted = sorted;
                unsorted.shuffle(&mut rng);
                unsorted
            };

            f(&mut unsorted);

            assert_eq!(unsorted, sorted);
        }
    }

    fn generate_random_tuples<R: Rng>(rng: &mut R) -> Vec<(u8, u8)> {
        use rand::distributions::Uniform;

        let size = match Uniform::new_inclusive(0u8, 2).sample(rng) {
            0 => Uniform::new_inclusive(128u8, u8::MAX).sample(rng) as usize,
            1 => Uniform::new_inclusive(u8::MAX as u16 + 1, u16::MAX).sample(rng) as usize,
            2 => Uniform::new_inclusive(u16::MAX as u32 + 1, (u16::MAX as u32) * 2).sample(rng)
                as usize,
            _ => unreachable!(),
        };

        let random_byte_distr = Uniform::new_inclusive(0u8, u8::MAX);

        (0..size)
            .map(|_| (random_byte_distr.sample(rng), random_byte_distr.sample(rng)))
            .collect()
    }

    fn sort_random_tuples<FU8, FU16, FU32>(mut fu8: FU8, mut fu16: FU16, mut fu32: FU32)
    where
        FU8: FnMut(&mut [(u8, u8)]),
        FU16: FnMut(&mut [(u8, u8)]),
        FU32: FnMut(&mut [(u8, u8)]),
    {
        let num_iterations = 100;

        for i in 0..num_iterations {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i);

            let mut unsorted = generate_random_tuples(&mut rng);
            let sorted = {
                let mut sorted = unsorted.clone();
                sorted.sort_unstable();
                sorted
            };

            if unsorted.len() <= u8::MAX as usize {
                fu8(&mut unsorted);
            } else if unsorted.len() <= u16::MAX as usize {
                fu16(&mut unsorted);
            } else {
                fu32(&mut unsorted);
            }

            assert!(unsorted == sorted);
        }
    }

    fn american_flag_sort_bytes<S, N>(bytes: &mut [u8])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        american_flag_sort::<1, { usize::MAX }, S, N, _, _, _>(
            bytes,
            &|byte: &u8, _| *byte,
            &|l: &u8, r: &u8, _| l.cmp(r),
            S::zero(),
            N::new(S::one()).unwrap(),
        );
    }

    fn american_flag_sort_bytes_u8(bytes: &mut [u8]) {
        assert!(bytes.len() <= u8::MAX as usize);
        american_flag_sort_bytes::<u8, NonZeroU8>(bytes);
    }

    fn american_flag_sort_bytes_u16(bytes: &mut [u8]) {
        assert!(bytes.len() <= u16::MAX as usize);
        american_flag_sort_bytes::<u16, NonZeroU16>(bytes);
    }

    fn american_flag_sort_bytes_u32(bytes: &mut [u8]) {
        assert!(bytes.len() <= u32::MAX as usize);
        american_flag_sort_bytes::<u32, NonZeroU32>(bytes);
    }

    fn american_flag_sort_tuples<S, N>(tuples: &mut [(u8, u8)])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        american_flag_sort::<1, { usize::MAX }, S, N, _, _, _>(
            tuples,
            &|tuple: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return tuple.0;
                }
                if lvl == S::one() {
                    return tuple.1;
                }
                unreachable!()
            },
            &|l: &(u8, u8), r: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return l.cmp(r);
                }
                if lvl == S::one() {
                    return l.1.cmp(&r.1);
                }
                unreachable!()
            },
            S::zero(),
            N::new(S::from_usize(2).unwrap()).unwrap(),
        );
    }

    fn american_flag_sort_tuples_u8(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u8::MAX as usize);
        american_flag_sort_tuples::<u8, NonZeroU8>(tuples)
    }

    fn american_flag_sort_tuples_u16(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u16::MAX as usize);
        american_flag_sort_tuples::<u16, NonZeroU16>(tuples)
    }

    fn american_flag_sort_tuples_u32(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u32::MAX as usize);
        american_flag_sort_tuples::<u32, NonZeroU32>(tuples)
    }

    #[test]
    fn american_flag_sort_unique_bytes() {
        sort_unique_bytes(american_flag_sort_bytes_u32);
    }

    #[test]
    fn american_flag_sort_unique_bytes_u8() {
        sort_unique_bytes_u8(american_flag_sort_bytes_u8);
    }

    #[test]
    fn american_flag_sort_random_bytes() {
        sort_random_bytes(
            american_flag_sort_bytes_u8,
            american_flag_sort_bytes_u16,
            american_flag_sort_bytes_u32,
        );
    }

    #[test]
    fn american_flag_sort_unique_tuples() {
        sort_unique_tuples(american_flag_sort_tuples_u8);
    }

    #[test]
    fn american_flag_sort_unique_tuples_u8() {
        sort_unique_tuples_u8(american_flag_sort_tuples_u8);
    }

    #[test]
    fn american_flag_sort_random_tuples() {
        sort_random_tuples(
            american_flag_sort_tuples_u8,
            american_flag_sort_tuples_u16,
            american_flag_sort_tuples_u32,
        );
    }

    fn ska_sort_bytes<S, N>(bytes: &mut [u8])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        ska_sort::<1, 1, S, N, _, _, _>(
            bytes,
            &|byte: &u8, _| *byte,
            &|l: &u8, r: &u8, _| l.cmp(r),
            S::zero(),
            N::new(S::one()).unwrap(),
        );
    }

    fn ska_sort_bytes_u8(bytes: &mut [u8]) {
        assert!(bytes.len() <= u8::MAX as usize);
        ska_sort_bytes::<u8, NonZeroU8>(bytes);
    }

    fn ska_sort_bytes_u16(bytes: &mut [u8]) {
        assert!(bytes.len() <= u16::MAX as usize);
        ska_sort_bytes::<u16, NonZeroU16>(bytes);
    }

    fn ska_sort_bytes_u32(bytes: &mut [u8]) {
        assert!(bytes.len() <= u32::MAX as usize);
        ska_sort_bytes::<u32, NonZeroU32>(bytes);
    }

    fn ska_sort_tuples<S, N>(tuples: &mut [(u8, u8)])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        ska_sort::<1, 1, S, N, _, _, _>(
            tuples,
            &|tuple: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return tuple.0;
                }
                if lvl == S::one() {
                    return tuple.1;
                }
                unreachable!()
            },
            &|l: &(u8, u8), r: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return l.cmp(r);
                }
                if lvl == S::one() {
                    return l.1.cmp(&r.1);
                }
                unreachable!()
            },
            S::zero(),
            N::new(S::from_usize(2).unwrap()).unwrap(),
        );
    }

    fn ska_sort_tuples_u8(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u8::MAX as usize);
        ska_sort_tuples::<u8, NonZeroU8>(tuples)
    }

    fn ska_sort_tuples_u16(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u16::MAX as usize);
        ska_sort_tuples::<u16, NonZeroU16>(tuples)
    }

    fn ska_sort_tuples_u32(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u32::MAX as usize);
        ska_sort_tuples::<u32, NonZeroU32>(tuples)
    }

    #[test]
    fn ska_sort_unique_bytes() {
        sort_unique_bytes(ska_sort_bytes_u32);
    }

    #[test]
    fn ska_sort_unique_bytes_u8() {
        sort_unique_bytes_u8(ska_sort_bytes_u8);
    }

    #[test]
    fn ska_sort_random_bytes() {
        sort_random_bytes(ska_sort_bytes_u8, ska_sort_bytes_u16, ska_sort_bytes_u32);
    }

    #[test]
    fn ska_sort_unique_tuples() {
        sort_unique_tuples(ska_sort_tuples_u8);
    }

    #[test]
    fn ska_sort_unique_tuples_u8() {
        sort_unique_tuples_u8(ska_sort_tuples_u8);
    }

    #[test]
    fn ska_sort_random_tuples() {
        sort_random_tuples(ska_sort_tuples_u8, ska_sort_tuples_u16, ska_sort_tuples_u32);
    }

    fn radix_sort_bytes<S, N>(bytes: &mut [u8])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        radix_sort::<S, N, _, _, _>(
            bytes,
            &|byte: &u8, _| *byte,
            &|l: &u8, r: &u8, _| l.cmp(r),
            N::new(S::one()).unwrap(),
        );
    }

    fn radix_sort_bytes_u8(bytes: &mut [u8]) {
        assert!(bytes.len() <= u8::MAX as usize);
        radix_sort_bytes::<u8, NonZeroU8>(bytes);
    }

    fn radix_sort_bytes_u16(bytes: &mut [u8]) {
        assert!(bytes.len() <= u16::MAX as usize);
        radix_sort_bytes::<u16, NonZeroU16>(bytes);
    }

    fn radix_sort_bytes_u32(bytes: &mut [u8]) {
        assert!(bytes.len() <= u32::MAX as usize);
        radix_sort_bytes::<u32, NonZeroU32>(bytes);
    }

    #[test]
    fn radix_sort_random_bytes() {
        sort_random_bytes(
            radix_sort_bytes_u8,
            radix_sort_bytes_u16,
            radix_sort_bytes_u32,
        );
    }

    fn radix_sort_tuples<S, N>(tuples: &mut [(u8, u8)])
    where
        S: Unsigned,
        N: NonZero<S>,
    {
        radix_sort::<S, N, _, _, _>(
            tuples,
            &|tuple: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return tuple.0;
                }
                if lvl == S::one() {
                    return tuple.1;
                }
                unreachable!()
            },
            &|l: &(u8, u8), r: &(u8, u8), lvl: S| {
                if lvl == S::zero() {
                    return l.cmp(r);
                }
                if lvl == S::one() {
                    return l.1.cmp(&r.1);
                }
                unreachable!()
            },
            N::new(S::from_usize(2).unwrap()).unwrap(),
        );
    }

    fn radix_sort_tuples_u8(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u8::MAX as usize);
        radix_sort_tuples::<u8, NonZeroU8>(tuples)
    }

    fn radix_sort_tuples_u16(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u16::MAX as usize);
        radix_sort_tuples::<u16, NonZeroU16>(tuples)
    }

    fn radix_sort_tuples_u32(tuples: &mut [(u8, u8)]) {
        assert!(tuples.len() <= u32::MAX as usize);
        radix_sort_tuples::<u32, NonZeroU32>(tuples)
    }

    #[test]
    fn radix_sort_random_tuples() {
        sort_random_tuples(
            radix_sort_tuples_u8,
            radix_sort_tuples_u16,
            radix_sort_tuples_u32,
        );
    }
}
