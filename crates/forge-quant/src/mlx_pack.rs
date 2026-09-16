//! MLX g128 bit-packing helpers (M0).
//!
//! Shared pack/unpack used by the sub-1-bit quantizers (M1). Layout matches
//! the MLX convention: codes packed LSB-first, `group` (default 128) weights
//! share one fp16 scale stored alongside (see [`scales_len`]).

use anyhow::{bail, Result};

/// Packed data bytes for `params` weights at `bits` (+ 1/2/4/8 only).
pub fn packed_len(params: usize, bits: u8) -> Result<usize> {
    match bits {
        1 | 2 | 4 | 8 => Ok((params * bits as usize).div_ceil(8)),
        b => bail!("mlx_pack supports 1, 2, 4 or 8 bits, got {}", b),
    }
}

/// Scale bytes: one fp16 scale per group.
pub fn scales_len(params: usize, group: usize) -> usize {
    params.div_ceil(group.max(1)) * 2
}

/// Pack integer codes (each < 2^bits) LSB-first into bytes.
///
/// M5: `chunks_exact` + slice writes (no per-push bounds checks on output).
pub fn pack_uniform(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    let per_byte = match bits {
        1 => 8,
        2 => 4,
        4 => 2,
        8 => return Ok(codes.to_vec()),
        b => bail!("mlx_pack supports 1, 2, 4 or 8 bits, got {}", b),
    };
    let mask = (1u8 << bits) - 1;
    let shift = bits as usize;
    let mut out = vec![0u8; codes.len().div_ceil(per_byte)];
    let (chunks, tail) = codes.split_at(codes.len() / per_byte * per_byte);
    for (o, chunk) in out.iter_mut().zip(chunks.chunks_exact(per_byte)) {
        let mut byte = 0u8;
        for (i, c) in chunk.iter().enumerate() {
            byte |= (c & mask) << (i * shift);
        }
        *o = byte;
    }
    if !tail.is_empty() {
        let mut byte = 0u8;
        for (i, c) in tail.iter().enumerate() {
            byte |= (c & mask) << (i * shift);
        }
        if let Some(last) = out.last_mut() {
            *last = byte;
        }
    }
    Ok(out)
}

/// Byte -> pre-expanded codes LUTs (M5): one table lookup replaces the
/// per-code shift+mask loop (8/4/2 codes per byte for 1/2/4-bit).
const fn build_lut1() -> [[u8; 8]; 256] {
    let mut t = [[0u8; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut i = 0usize;
        while i < 8 {
            t[b][i] = ((b >> i) & 1) as u8;
            i += 1;
        }
        b += 1;
    }
    t
}
const fn build_lut2() -> [[u8; 4]; 256] {
    let mut t = [[0u8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut i = 0usize;
        while i < 4 {
            t[b][i] = ((b >> (i * 2)) & 3) as u8;
            i += 1;
        }
        b += 1;
    }
    t
}
const fn build_lut4() -> [[u8; 2]; 256] {
    let mut t = [[0u8; 2]; 256];
    let mut b = 0usize;
    while b < 256 {
        t[b][0] = (b & 15) as u8;
        t[b][1] = (b >> 4) as u8;
        b += 1;
    }
    t
}
static LUT1: [[u8; 8]; 256] = build_lut1();
static LUT2: [[u8; 4]; 256] = build_lut2();
static LUT4: [[u8; 2]; 256] = build_lut4();

/// Unpack LSB-first bytes back into `n` codes.
pub fn unpack_uniform(packed: &[u8], bits: u8, n: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; n];
    match bits {
        1 => {
            let (full, tail) = out.split_at_mut(n / 8 * 8);
            for (o, b) in full.chunks_exact_mut(8).zip(packed.iter()) {
                o.copy_from_slice(&LUT1[*b as usize]);
            }
            if !tail.is_empty() {
                if let Some(b) = packed.get(full.len() / 8) {
                    tail.copy_from_slice(&LUT1[*b as usize][..tail.len()]);
                } else {
                    bail!("packed too short: need {} codes, have {} bytes", n, packed.len());
                }
            }
        }
        2 => {
            let (full, tail) = out.split_at_mut(n / 4 * 4);
            for (o, b) in full.chunks_exact_mut(4).zip(packed.iter()) {
                o.copy_from_slice(&LUT2[*b as usize]);
            }
            if !tail.is_empty() {
                if let Some(b) = packed.get(full.len() / 4) {
                    tail.copy_from_slice(&LUT2[*b as usize][..tail.len()]);
                } else {
                    bail!("packed too short: need {} codes, have {} bytes", n, packed.len());
                }
            }
        }
        4 => {
            let (full, tail) = out.split_at_mut(n / 2 * 2);
            for (o, b) in full.chunks_exact_mut(2).zip(packed.iter()) {
                o.copy_from_slice(&LUT4[*b as usize]);
            }
            if !tail.is_empty() {
                if let Some(b) = packed.get(full.len() / 2) {
                    tail.copy_from_slice(&LUT4[*b as usize][..tail.len()]);
                } else {
                    bail!("packed too short: need {} codes, have {} bytes", n, packed.len());
                }
            }
        }
        8 => return Ok(packed.iter().cloned().take(n).collect()),
        b => bail!("mlx_pack supports 1, 2, 4 or 8 bits, got {}", b),
    }
    // Short-input guard for the exact-fit path: zip stops at the shorter
    // side, so verify enough bytes were consumed.
    let need = n.div_ceil(match bits { 1 => 8, 2 => 4, _ => 2 });
    if packed.len() < need {
        bail!("packed too short: need {} codes, have {} bytes", n, packed.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_len_is_exact() {
        assert_eq!(packed_len(1024, 1).unwrap(), 128);
        assert_eq!(packed_len(1024, 2).unwrap(), 256);
        assert_eq!(packed_len(1024, 4).unwrap(), 512);
        assert_eq!(packed_len(1024, 8).unwrap(), 1024);
        assert!(packed_len(1024, 3).is_err());
    }

    #[test]
    fn scales_len_g128() {
        assert_eq!(scales_len(1024, 128), 16);
        assert_eq!(scales_len(100, 128), 2);
    }

    #[test]
    fn roundtrip_one_two_four_bit() {
        for bits in [1u8, 2, 4] {
            let levels = 1u8 << bits;
            let codes: Vec<u8> = (0..300).map(|i| (i % levels as usize) as u8).collect();
            let packed = pack_uniform(&codes, bits).unwrap();
            assert_eq!(packed.len(), packed_len(codes.len(), bits).unwrap());
            let back = unpack_uniform(&packed, bits, codes.len()).unwrap();
            assert_eq!(back, codes);
        }
    }
}
