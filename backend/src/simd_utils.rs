pub fn normalize_nonnegative_in_place(values: &mut [f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { normalize_nonnegative_avx512(values) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { normalize_nonnegative_avx2(values) };
        }
    }
    normalize_nonnegative_scalar(values)
}

fn normalize_nonnegative_scalar(values: &mut [f32]) -> f32 {
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = value.max(0.0);
        sum += *value;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for value in values.iter_mut() {
            *value *= inv;
        }
    }
    sum
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn normalize_nonnegative_avx2(values: &mut [f32]) -> f32 {
    let len = values.len();
    let zero = _mm256_setzero_ps();
    let mut acc = _mm256_setzero_ps();
    let ptr = values.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= len {
        let v = unsafe { _mm256_loadu_ps(ptr.add(i)) };
        let clamped = _mm256_max_ps(v, zero);
        unsafe { _mm256_storeu_ps(ptr.add(i), clamped) };
        acc = _mm256_add_ps(acc, clamped);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().copied().sum::<f32>();
    while i < len {
        let value = unsafe { (*ptr.add(i)).max(0.0) };
        unsafe { *ptr.add(i) = value };
        sum += value;
        i += 1;
    }
    if sum > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum);
        let mut j = 0usize;
        while j + 8 <= len {
            let v = unsafe { _mm256_loadu_ps(ptr.add(j)) };
            unsafe { _mm256_storeu_ps(ptr.add(j), _mm256_mul_ps(v, inv)) };
            j += 8;
        }
        while j < len {
            unsafe { *ptr.add(j) /= sum };
            j += 1;
        }
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn normalize_nonnegative_avx512(values: &mut [f32]) -> f32 {
    let len = values.len();
    let zero = _mm512_setzero_ps();
    let mut acc = _mm512_setzero_ps();
    let ptr = values.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= len {
        let v = unsafe { _mm512_loadu_ps(ptr.add(i)) };
        let clamped = _mm512_max_ps(v, zero);
        unsafe { _mm512_storeu_ps(ptr.add(i), clamped) };
        acc = _mm512_add_ps(acc, clamped);
        i += 16;
    }
    let mut lanes = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().copied().sum::<f32>();
    while i < len {
        let value = unsafe { (*ptr.add(i)).max(0.0) };
        unsafe { *ptr.add(i) = value };
        sum += value;
        i += 1;
    }
    if sum > 0.0 {
        let inv = _mm512_set1_ps(1.0 / sum);
        let mut j = 0usize;
        while j + 16 <= len {
            let v = unsafe { _mm512_loadu_ps(ptr.add(j)) };
            unsafe { _mm512_storeu_ps(ptr.add(j), _mm512_mul_ps(v, inv)) };
            j += 16;
        }
        while j < len {
            unsafe { *ptr.add(j) /= sum };
            j += 1;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::normalize_nonnegative_in_place;

    #[test]
    fn normalize_clamps_and_scales() {
        let mut values = vec![0.5, -1.0, 1.5, 0.0];
        let sum = normalize_nonnegative_in_place(&mut values);
        assert!((sum - 2.0).abs() < 1e-6);
        assert_eq!(values[1], 0.0);
        let total: f32 = values.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_sum_stays_zero() {
        let mut values = vec![-1.0, 0.0, -3.0];
        let sum = normalize_nonnegative_in_place(&mut values);
        assert_eq!(sum, 0.0);
        assert_eq!(values, vec![0.0, 0.0, 0.0]);
    }
}
