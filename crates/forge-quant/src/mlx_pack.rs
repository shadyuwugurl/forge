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
pub fn pack_uniform(codes: &[u8], bits: u8) -> Result<Vec<u8>> {
    let per_byte = match bits {
        1 => 8,
        2 => 4,
        4 => 2,
        8 => return Ok(codes.to_vec()),
        b => bail!("mlx_pack supports 1, 2, 4 or 8 bits, got {}", b),
    };
    let mask = (1u8 << bits) - 1;
    let mut out = Vec::with_capacity(codes.len() / per_byte + 1);
    for chunk in codes.chunks(per_byte) {
        let mut byte = 0u8;
        for (i, c) in chunk.iter().enumerate() {
            byte |= (c & mask) << (i * bits as usize);
        }
        out.push(byte);
    }
    Ok(out)
}

/// Unpack LSB-first bytes back into `n` codes.
pub fn unpack_uniform(packed: &[u8], bits: u8, n: usize) -> Result<Vec<u8>> {
    let per_byte = match bits {
        1 => 8,
        2 => 4,
        4 => 2,
        8 => return Ok(packed.iter().cloned().take(n).collect()),
        b => bail!("mlx_pack supports 1, 2, 4 or 8 bits, got {}", b),
    };
    let mask = (1u8 << bits) - 1;
    let mut out = Vec::with_capacity(n);
    for byte in packed {
        for i in 0..per_byte {
            if out.len() >= n {
                break;
            }
            out.push((byte >> (i * bits as usize)) & mask);
        }
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
