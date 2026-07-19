use std::{
    hash::{Hash, Hasher},
    iter::{ExactSizeIterator, FusedIterator},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Shl, ShlAssign, Shr, ShrAssign,
        Sub, SubAssign,
    },
};

/// A compact bitset for tracking slot indices in CSE.
///
/// Uses stack-optimized representations for small sizes (up to 32, 64, and 128
/// bits) and falls back to a heap-allocated `Vec<u64>` for larger sets.
#[derive(Debug, Clone, Eq)]
pub enum IndexSet {
    /// Up to 32 bits stored inline.
    Small(u32),
    /// 33–64 bits stored inline.
    Medium(u64),
    /// 65–128 bits stored inline.
    Large(u128),
    /// More than 128 bits, stored as a heap-allocated vector of 64-bit chunks.
    Heap(Vec<u64>),
}

impl Default for IndexSet {
    #[inline]
    fn default() -> Self {
        IndexSet::Small(0)
    }
}

impl IndexSet {
    /// Create a set containing exactly one slot index.
    #[inline]
    pub fn singleton(slot: usize) -> Self {
        if slot < 32 {
            IndexSet::Small(1u32 << slot)
        } else if slot < 64 {
            IndexSet::Medium(1u64 << slot)
        } else if slot < 128 {
            IndexSet::Large(1u128 << slot)
        } else {
            let chunks = (slot / 64) + 1;
            let mut vec = vec![0u64; chunks];
            vec[slot / 64] |= 1u64 << (slot % 64);
            IndexSet::Heap(vec)
        }
    }

    /// Insert or remove a slot index.
    ///
    /// Automatically promotes the representation when the slot exceeds the
    /// current variant's capacity.
    pub fn insert(&mut self, slot: usize, value: bool) {
        match self {
            IndexSet::Small(bits) => {
                if slot < 32 {
                    if value {
                        *bits |= 1u32 << slot;
                    } else {
                        *bits &= !(1u32 << slot);
                    }
                } else if value {
                    if slot < 64 {
                        *self = IndexSet::Medium((*bits as u64) | (1u64 << slot));
                    } else if slot < 128 {
                        *self = IndexSet::Large((*bits as u128) | (1u128 << slot));
                    } else {
                        let chunks = (slot / 64) + 1;
                        let mut vec = vec![0u64; chunks];
                        vec[0] = *bits as u64;
                        vec[slot / 64] |= 1u64 << (slot % 64);
                        *self = IndexSet::Heap(vec);
                    }
                }
            }
            IndexSet::Medium(bits) => {
                if slot < 64 {
                    if value {
                        *bits |= 1u64 << slot;
                    } else {
                        *bits &= !(1u64 << slot);
                    }
                } else if value {
                    if slot < 128 {
                        *self = IndexSet::Large((*bits as u128) | (1u128 << slot));
                    } else {
                        let chunks = (slot / 64) + 1;
                        let mut vec = vec![0u64; chunks];
                        vec[0] = *bits;
                        vec[slot / 64] |= 1u64 << (slot % 64);
                        *self = IndexSet::Heap(vec);
                    }
                }
            }
            IndexSet::Large(bits) => {
                if slot < 128 {
                    if value {
                        *bits |= 1u128 << slot;
                    } else {
                        *bits &= !(1u128 << slot);
                    }
                } else if value {
                    let chunks = (slot / 64) + 1;
                    let mut vec = vec![0u64; chunks];
                    vec[0] = *bits as u64;
                    vec[1] = (*bits >> 64) as u64;
                    vec[slot / 64] |= 1u64 << (slot % 64);
                    *self = IndexSet::Heap(vec);
                }
            }
            IndexSet::Heap(vec) => {
                let idx = slot / 64;
                let bit = slot % 64;
                if value {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, 0);
                    }
                    vec[idx] |= 1u64 << bit;
                } else {
                    if idx < vec.len() {
                        vec[idx] &= !(1u64 << bit);
                    }
                }
            }
        }
    }

    /// Returns `true` if the two sets have no elements in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (IndexSet::Small(a), IndexSet::Small(b)) => (a & b) == 0,
            (IndexSet::Small(a), IndexSet::Medium(b)) => ((*a as u64) & b) == 0,
            (IndexSet::Medium(a), IndexSet::Small(b)) => (a & (*b as u64)) == 0,
            (IndexSet::Small(a), IndexSet::Large(b)) => ((*a as u128) & b) == 0,
            (IndexSet::Large(a), IndexSet::Small(b)) => (a & (*b as u128)) == 0,
            (IndexSet::Medium(a), IndexSet::Medium(b)) => (a & b) == 0,
            (IndexSet::Medium(a), IndexSet::Large(b)) => ((*a as u128) & b) == 0,
            (IndexSet::Large(a), IndexSet::Medium(b)) => (a & (*b as u128)) == 0,
            (IndexSet::Large(a), IndexSet::Large(b)) => (a & b) == 0,
            (IndexSet::Heap(a), IndexSet::Heap(b)) => {
                let min_len = a.len().min(b.len());
                for i in 0..min_len {
                    if (a[i] & b[i]) != 0 {
                        return false;
                    }
                }
                true
            }
            (IndexSet::Small(a), IndexSet::Heap(b)) => {
                b.first().is_none_or(|&x| ((*a as u64) & x) == 0)
            }
            (IndexSet::Heap(a), IndexSet::Small(b)) => {
                a.first().is_none_or(|&x| (x & (*b as u64)) == 0)
            }
            (IndexSet::Medium(a), IndexSet::Heap(b)) => b.first().is_none_or(|&x| (a & x) == 0),
            (IndexSet::Heap(a), IndexSet::Medium(b)) => a.first().is_none_or(|&x| (x & b) == 0),
            (IndexSet::Large(a), IndexSet::Heap(b)) => {
                if b.is_empty() {
                    return true;
                }
                if (b[0] & (*a as u64)) != 0 {
                    return false;
                }
                if b.len() > 1 && (b[1] & ((*a >> 64) as u64)) != 0 {
                    return false;
                }
                true
            }
            (IndexSet::Heap(a), IndexSet::Large(b)) => {
                if a.is_empty() {
                    return true;
                }
                if (a[0] & (*b as u64)) != 0 {
                    return false;
                }
                if a.len() > 1 && (a[1] & ((*b >> 64) as u64)) != 0 {
                    return false;
                }
                true
            }
        }
    }

    /// Returns `true` if the given slot index is present in the set.
    #[inline]
    pub fn contains(&self, slot: usize) -> bool {
        match self {
            IndexSet::Small(bits) => slot < 32 && (*bits & (1u32 << slot)) != 0,
            IndexSet::Medium(bits) => slot < 64 && (*bits & (1u64 << slot)) != 0,
            IndexSet::Large(bits) => slot < 128 && (*bits & (1u128 << slot)) != 0,
            IndexSet::Heap(vec) => {
                let idx = slot / 64;
                let bit = slot % 64;
                idx < vec.len() && (vec[idx] & (1u64 << bit)) != 0
            }
        }
    }

    /// Iterate over all slot indices present in the set, in ascending order.
    #[inline]
    pub fn iter<'a>(&'a self) -> IndexSetIter<'a> {
        let max = self.max_chunks().saturating_sub(1);
        IndexSetIter {
            inner: self,
            front_chunk: 0,
            front_bits: self.get_first_chunk(),
            back_chunk: max,
            back_bits: if max == 0 {
                self.get_first_chunk()
            } else {
                match self {
                    IndexSet::Large(b) => (*b >> 64) as u64,
                    IndexSet::Heap(v) => v.last().copied().unwrap_or(0),
                    _ => unreachable!("Small/Medium have only one chunk"),
                }
            },
            remaining: self.count_ones(),
        }
    }

    #[inline]
    fn get_first_chunk(&self) -> u64 {
        match self {
            IndexSet::Small(b) => *b as u64,
            IndexSet::Medium(b) => *b,
            IndexSet::Large(b) => *b as u64,
            IndexSet::Heap(v) => v.first().copied().unwrap_or(0),
        }
    }

    #[inline]
    fn max_chunks(&self) -> usize {
        match self {
            IndexSet::Small(_) | IndexSet::Medium(_) => 1,
            IndexSet::Large(_) => 2,
            IndexSet::Heap(vec) => vec.len(),
        }
    }

    /// Returns the number of slot indices in the set (population count).
    #[inline]
    pub fn count_ones(&self) -> usize {
        match self {
            IndexSet::Small(bits) => bits.count_ones() as usize,
            IndexSet::Medium(bits) => bits.count_ones() as usize,
            IndexSet::Large(bits) => bits.count_ones() as usize,
            IndexSet::Heap(vec) => vec.iter().map(|&x| x.count_ones() as usize).sum(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            IndexSet::Small(bits) => *bits == 0,
            IndexSet::Medium(bits) => *bits == 0,
            IndexSet::Large(bits) => *bits == 0,
            IndexSet::Heap(vec) => vec.iter().all(|&x| x == 0),
        }
    }

    /// Shrink to the smallest variant that can hold the current bits.
    #[inline]
    pub fn minimized(mut self) -> Self {
        self.minimize();
        self
    }

    /// Shrink to the smallest variant that can hold the current bits.
    pub fn minimize(&mut self) {
        *self = match std::mem::take(self) {
            IndexSet::Heap(mut vec) => {
                let last = vec.iter().rposition(|&x| x != 0);
                match last {
                    None => IndexSet::Small(0),
                    Some(0) => {
                        let v = vec[0];
                        if v as u32 as u64 == v {
                            IndexSet::Small(v as u32)
                        } else {
                            IndexSet::Medium(v)
                        }
                    }
                    Some(1) => IndexSet::Large((vec[0] as u128) | ((vec[1] as u128) << 64)),
                    _ => {
                        vec.truncate(last.unwrap() + 1);
                        IndexSet::Heap(vec)
                    }
                }
            }
            IndexSet::Large(0) => IndexSet::Small(0),
            IndexSet::Large(v) if (v as u64 as u128) == v => {
                let low = v as u64;
                if low as u32 as u64 == low {
                    IndexSet::Small(low as u32)
                } else {
                    IndexSet::Medium(low)
                }
            }
            IndexSet::Medium(0) => IndexSet::Small(0),
            IndexSet::Medium(v) if (v as u32 as u64) == v => IndexSet::Small(v as u32),
            other => other,
        };
    }
}

impl From<IndexSet> for Vec<u64> {
    fn from(set: IndexSet) -> Self {
        match set {
            IndexSet::Small(bits) => vec![bits as u64],
            IndexSet::Medium(bits) => vec![bits],
            IndexSet::Large(bits) => vec![bits as u64, (bits >> 64) as u64],
            IndexSet::Heap(vec) => vec,
        }
    }
}

impl PartialEq for IndexSet {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (IndexSet::Small(a), IndexSet::Small(b)) => a == b,
            (IndexSet::Medium(a), IndexSet::Medium(b)) => a == b,
            (IndexSet::Large(a), IndexSet::Large(b)) => a == b,
            _ => self.iter().eq(other.iter()),
        }
    }
}

impl PartialOrd for IndexSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (IndexSet::Small(a), IndexSet::Small(b)) => a.cmp(b),
            (IndexSet::Medium(a), IndexSet::Medium(b)) => a.cmp(b),
            (IndexSet::Large(a), IndexSet::Large(b)) => a.cmp(b),
            // Compare from the highest bit down — same as integer cmp.
            _ => self.iter().rev().cmp(other.iter().rev()),
        }
    }
}

impl Hash for IndexSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.count_ones().hash(state);

        for idx in self.iter() {
            idx.hash(state);
        }
    }
}

impl BitOr for IndexSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        match (self, rhs) {
            (IndexSet::Small(a), IndexSet::Small(b)) => IndexSet::Small(a | b),
            (IndexSet::Small(a), IndexSet::Medium(b)) => IndexSet::Medium((a as u64) | b),
            (IndexSet::Medium(a), IndexSet::Small(b)) => IndexSet::Medium(a | (b as u64)),
            (IndexSet::Small(a), IndexSet::Large(b)) => IndexSet::Large((a as u128) | b),
            (IndexSet::Large(a), IndexSet::Small(b)) => IndexSet::Large(a | (b as u128)),
            (IndexSet::Small(a), IndexSet::Heap(mut b)) => {
                if b.is_empty() {
                    b.push(a as u64);
                } else {
                    b[0] |= a as u64;
                }
                IndexSet::Heap(b)
            }
            (IndexSet::Heap(mut a), IndexSet::Small(b)) => {
                if a.is_empty() {
                    a.push(b as u64);
                } else {
                    a[0] |= b as u64;
                }
                IndexSet::Heap(a)
            }
            (IndexSet::Medium(a), IndexSet::Medium(b)) => IndexSet::Medium(a | b),
            (IndexSet::Medium(a), IndexSet::Large(b)) => IndexSet::Large((a as u128) | b),
            (IndexSet::Large(a), IndexSet::Medium(b)) => IndexSet::Large(a | (b as u128)),
            (IndexSet::Medium(a), IndexSet::Heap(mut b)) => {
                if b.is_empty() {
                    b.push(a);
                } else {
                    b[0] |= a;
                }
                IndexSet::Heap(b)
            }
            (IndexSet::Heap(mut a), IndexSet::Medium(b)) => {
                if a.is_empty() {
                    a.push(b);
                } else {
                    a[0] |= b;
                }
                IndexSet::Heap(a)
            }
            (IndexSet::Large(a), IndexSet::Large(b)) => IndexSet::Large(a | b),
            (IndexSet::Large(a), IndexSet::Heap(mut b)) => {
                if b.len() < 2 {
                    b.resize(2, 0);
                }
                b[0] |= a as u64;
                b[1] |= (a >> 64) as u64;
                IndexSet::Heap(b)
            }
            (IndexSet::Heap(mut a), IndexSet::Large(b)) => {
                if a.len() < 2 {
                    a.resize(2, 0);
                }
                a[0] |= b as u64;
                a[1] |= (b >> 64) as u64;
                IndexSet::Heap(a)
            }
            (IndexSet::Heap(mut a), IndexSet::Heap(b)) => {
                if a.len() < b.len() {
                    a.resize(b.len(), 0);
                }
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x |= *y;
                }
                IndexSet::Heap(a)
            }
        }
    }
}

impl BitOrAssign for IndexSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) | rhs;
    }
}

impl BitAnd for IndexSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        match (self, rhs) {
            (IndexSet::Small(a), IndexSet::Small(b)) => IndexSet::Small(a & b),
            (IndexSet::Small(a), IndexSet::Medium(b)) => IndexSet::Small((a as u64 & b) as u32),
            (IndexSet::Medium(a), IndexSet::Small(b)) => IndexSet::Small((a & (b as u64)) as u32),
            (IndexSet::Small(a), IndexSet::Large(b)) => IndexSet::Small((a as u128 & b) as u32),
            (IndexSet::Large(a), IndexSet::Small(b)) => IndexSet::Small((a & (b as u128)) as u32),
            (IndexSet::Small(a), IndexSet::Heap(b)) => {
                if b.is_empty() {
                    IndexSet::default()
                } else {
                    IndexSet::Small((a as u64 & b[0]) as u32)
                }
            }
            (IndexSet::Heap(a), IndexSet::Small(b)) => {
                if a.is_empty() {
                    IndexSet::default()
                } else {
                    IndexSet::Small((a[0] & (b as u64)) as u32)
                }
            }
            (IndexSet::Medium(a), IndexSet::Medium(b)) => IndexSet::Medium(a & b),
            (IndexSet::Medium(a), IndexSet::Large(b)) => IndexSet::Medium((a as u128 & b) as u64),
            (IndexSet::Large(a), IndexSet::Medium(b)) => IndexSet::Medium((a & (b as u128)) as u64),
            (IndexSet::Medium(a), IndexSet::Heap(b)) => {
                if b.is_empty() {
                    IndexSet::Medium(0)
                } else {
                    IndexSet::Medium(a & b[0])
                }
            }
            (IndexSet::Heap(a), IndexSet::Medium(b)) => {
                if a.is_empty() {
                    IndexSet::Medium(0)
                } else {
                    IndexSet::Medium(a[0] & b)
                }
            }
            (IndexSet::Large(a), IndexSet::Large(b)) => IndexSet::Large(a & b),
            (IndexSet::Large(a), IndexSet::Heap(b)) => {
                let lo = b.first().map(|&x| a as u64 & x).unwrap_or(0);
                let hi = b.get(1).map(|&x| (a >> 64) as u64 & x).unwrap_or(0);
                if hi == 0 {
                    IndexSet::Medium(lo)
                } else {
                    IndexSet::Large((lo as u128) | ((hi as u128) << 64))
                }
            }
            (IndexSet::Heap(a), IndexSet::Large(b)) => {
                let lo = a.first().map(|&x| x & (b as u64)).unwrap_or(0);
                let hi = a.get(1).map(|&x| x & ((b >> 64) as u64)).unwrap_or(0);
                if hi == 0 {
                    IndexSet::Medium(lo)
                } else {
                    IndexSet::Large((lo as u128) | ((hi as u128) << 64))
                }
            }
            (IndexSet::Heap(mut a), IndexSet::Heap(b)) => {
                let min_len = a.len().min(b.len());
                a.truncate(min_len);
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x &= *y;
                }
                IndexSet::Heap(a)
            }
        }
        .minimized()
    }
}

impl BitAndAssign for IndexSet {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) & rhs;
    }
}

#[inline]
fn heap_shl(vec: Vec<u64>, rhs: usize) -> IndexSet {
    let chunk_shift = rhs / 64;
    let bit_shift = rhs % 64;
    let mut new_vec = Vec::with_capacity(chunk_shift + vec.len() + 1);
    new_vec.extend(std::iter::repeat_n(0u64, chunk_shift));
    if bit_shift == 0 {
        new_vec.extend(vec);
    } else {
        let mut carry = 0u64;
        for v in vec {
            let val = (v << bit_shift) | carry;
            carry = v >> (64 - bit_shift);
            new_vec.push(val);
        }
        if carry != 0 {
            new_vec.push(carry);
        }
    }
    IndexSet::Heap(new_vec)
}

impl Shl<usize> for IndexSet {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self {
        if rhs == 0 || self.is_empty() {
            return self;
        }
        match self {
            IndexSet::Small(bits) => {
                let val = bits as u128;
                if (val.leading_zeros() as usize) < rhs {
                    heap_shl(vec![bits as u64], rhs)
                } else {
                    let shifted = val << rhs;
                    if shifted as u32 as u128 == shifted {
                        IndexSet::Small(shifted as u32)
                    } else if shifted as u64 as u128 == shifted {
                        IndexSet::Medium(shifted as u64)
                    } else {
                        IndexSet::Large(shifted)
                    }
                }
            }
            IndexSet::Medium(bits) => {
                let val = bits as u128;
                if (val.leading_zeros() as usize) < rhs {
                    heap_shl(vec![bits], rhs)
                } else {
                    let shifted = val << rhs;
                    if shifted as u64 as u128 == shifted {
                        IndexSet::Medium(shifted as u64)
                    } else {
                        IndexSet::Large(shifted)
                    }
                }
            }
            IndexSet::Large(bits) => {
                if (bits.leading_zeros() as usize) < rhs {
                    let lo = bits as u64;
                    let hi = (bits >> 64) as u64;
                    let vec = if hi != 0 { vec![lo, hi] } else { vec![lo] };
                    heap_shl(vec, rhs)
                } else {
                    IndexSet::Large(bits << rhs)
                }
            }
            IndexSet::Heap(vec) => heap_shl(vec, rhs),
        }
    }
}

impl ShlAssign<usize> for IndexSet {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = std::mem::take(self) << rhs;
    }
}

impl Shr<usize> for IndexSet {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self {
        if rhs == 0 || self.is_empty() {
            return self;
        }
        match self {
            IndexSet::Small(bits) => {
                if rhs >= 32 {
                    IndexSet::Small(0)
                } else {
                    IndexSet::Small(bits >> rhs)
                }
            }
            IndexSet::Medium(bits) => {
                if rhs >= 64 {
                    IndexSet::Small(0)
                } else {
                    IndexSet::Medium(bits >> rhs)
                }
            }
            IndexSet::Large(bits) => {
                if rhs >= 128 {
                    IndexSet::Small(0)
                } else {
                    IndexSet::Large(bits >> rhs)
                }
            }
            IndexSet::Heap(vec) => {
                let chunk_shift = rhs / 64;
                let bit_shift = rhs % 64;
                if chunk_shift >= vec.len() {
                    return IndexSet::Small(0);
                }
                let remaining = &vec[chunk_shift..];
                if bit_shift == 0 {
                    IndexSet::Heap(remaining.to_vec())
                } else {
                    let mut new_vec = Vec::with_capacity(remaining.len());
                    for i in 0..remaining.len() {
                        let mut val = remaining[i] >> bit_shift;
                        if i + 1 < remaining.len() {
                            val |= remaining[i + 1] << (64 - bit_shift);
                        }
                        new_vec.push(val);
                    }
                    IndexSet::Heap(new_vec)
                }
            }
        }
        .minimized()
    }
}

impl ShrAssign<usize> for IndexSet {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = std::mem::take(self) >> rhs;
    }
}

/// An iterator over the slot indices contained in an [`IndexSet`].
///
/// Produces indices in ascending order.
pub struct IndexSetIter<'a> {
    inner: &'a IndexSet,
    front_chunk: usize,
    front_bits: u64,
    back_chunk: usize,
    back_bits: u64,
    remaining: usize,
}

impl IndexSetIter<'_> {
    #[inline]
    fn chunk_bits(&self, chunk: usize) -> u64 {
        match self.inner {
            IndexSet::Small(bits) if chunk == 0 => *bits as u64,
            IndexSet::Medium(bits) if chunk == 0 => *bits,
            IndexSet::Large(bits) if chunk == 0 => *bits as u64,
            IndexSet::Large(bits) if chunk == 1 => (*bits >> 64) as u64,
            IndexSet::Heap(vec) => vec.get(chunk).copied().unwrap_or(0),
            _ => 0,
        }
    }
}

impl<'a> Iterator for IndexSetIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.front_bits != 0 {
                let tz = self.front_bits.trailing_zeros() as usize;
                let mask = 1u64 << tz;
                self.front_bits &= !mask;
                if self.front_chunk == self.back_chunk {
                    self.back_bits &= !mask;
                }
                self.remaining -= 1;
                return Some(self.front_chunk * 64 + tz);
            }

            if self.remaining == 0 {
                return None;
            }

            self.front_chunk += 1;
            self.front_bits = self.chunk_bits(self.front_chunk);
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> DoubleEndedIterator for IndexSetIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.back_bits != 0 {
                let lz = self.back_bits.ilog2() as usize;
                let mask = 1u64 << lz;
                self.back_bits &= !mask;
                if self.front_chunk == self.back_chunk {
                    self.front_bits &= !mask;
                }
                self.remaining -= 1;
                return Some(self.back_chunk * 64 + lz);
            }

            if self.remaining == 0 {
                return None;
            }

            if self.back_chunk == 0 {
                self.remaining = 0;
                return None;
            }
            self.back_chunk -= 1;
            self.back_bits = self.chunk_bits(self.back_chunk);
        }
    }
}

impl<'a> ExactSizeIterator for IndexSetIter<'a> {}

impl<'a> FusedIterator for IndexSetIter<'a> {}

/// A type around [`IndexSet`] providing BigUint-like arithmetic.
///
/// Treats the bitset as an unsigned integer:
/// `Add` / `AddAssign` perform binary addition with carry propagation;
/// `Sub` / `SubAssign` perform binary subtraction with borrow (panics on underflow).
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArithIndexSet(pub IndexSet);

impl From<IndexSet> for ArithIndexSet {
    #[inline]
    fn from(set: IndexSet) -> Self {
        ArithIndexSet(set)
    }
}

impl From<ArithIndexSet> for IndexSet {
    #[inline]
    fn from(set: ArithIndexSet) -> Self {
        set.0
    }
}

impl From<u8> for ArithIndexSet {
    #[inline]
    fn from(value: u8) -> Self {
        ArithIndexSet(IndexSet::Small(value as u32))
    }
}

impl From<u16> for ArithIndexSet {
    #[inline]
    fn from(value: u16) -> Self {
        ArithIndexSet(IndexSet::Small(value as u32))
    }
}

impl From<u32> for ArithIndexSet {
    #[inline]
    fn from(value: u32) -> Self {
        ArithIndexSet(IndexSet::Small(value))
    }
}

impl From<u64> for ArithIndexSet {
    #[inline]
    fn from(value: u64) -> Self {
        ArithIndexSet(IndexSet::Medium(value))
    }
}

impl From<u128> for ArithIndexSet {
    #[inline]
    fn from(value: u128) -> Self {
        ArithIndexSet(IndexSet::Large(value))
    }
}

impl From<usize> for ArithIndexSet {
    #[inline]
    #[cfg(target_pointer_width = "32")]
    fn from(value: usize) -> Self {
        ArithIndexSet(IndexSet::Small(value as u32))
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    fn from(value: usize) -> Self {
        ArithIndexSet(IndexSet::Medium(value as u64))
    }
}

impl std::ops::Deref for ArithIndexSet {
    type Target = IndexSet;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for ArithIndexSet {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for ArithIndexSet {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }
}

impl AddAssign for ArithIndexSet {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) + rhs;
    }
}

impl ArithIndexSet {
    /// Integer addition with overflow. Returns `(sum, overflowed)`
    ///
    /// `overflowed` is `true` if the result promoted to a larger variant.
    #[must_use]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (a, b) = (self.0, rhs.0);
        let (inner, overflow) =
            match (a, b) {
                (IndexSet::Small(a), IndexSet::Small(b)) => add_small(a, b),
                (IndexSet::Small(b), IndexSet::Medium(a))
                | (IndexSet::Medium(a), IndexSet::Small(b)) => add_medium(a, b as u64),
                (IndexSet::Medium(a), IndexSet::Medium(b)) => add_medium(a, b),
                (IndexSet::Small(b), IndexSet::Large(a))
                | (IndexSet::Large(a), IndexSet::Small(b)) => add_large(a, b as u128),
                (IndexSet::Medium(b), IndexSet::Large(a))
                | (IndexSet::Large(a), IndexSet::Medium(b)) => add_large(a, b as u128),
                (IndexSet::Large(a), IndexSet::Large(b)) => add_large(a, b),
                (IndexSet::Small(b), IndexSet::Heap(a))
                | (IndexSet::Heap(a), IndexSet::Small(b)) => add_heap(a, b as u64),
                (IndexSet::Medium(b), IndexSet::Heap(a))
                | (IndexSet::Heap(a), IndexSet::Medium(b)) => add_heap(a, b),
                (a, b) => {
                    let a: Vec<u64> = a.into();
                    let b: Vec<u64> = b.into();
                    let max_len = a.len().max(b.len());
                    let mut result = Vec::with_capacity(max_len + 1);
                    let mut carry: bool = false;
                    for i in 0..max_len {
                        let av = a.get(i).copied().unwrap_or(0);
                        let bv = b.get(i).copied().unwrap_or(0);
                        let (sum, ov1) = av.overflowing_add(bv);
                        let (sum2, ov2) = sum.overflowing_add(u64::from(carry));
                        result.push(sum2);
                        carry = ov1 || ov2;
                    }
                    if carry {
                        result.push(1);
                    }
                    (IndexSet::Heap(result), carry)
                }
            };
        (ArithIndexSet(inner), overflow)
    }

    /// Checked addition. Returns `None` if the result would overflow the
    /// representation (would require promotion to a larger variant).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let (result, overflow) = self.overflowing_add(rhs);
        if overflow {
            None
        } else {
            Some(ArithIndexSet(result.0.minimized()))
        }
    }
}

#[inline]
fn add_small(a: u32, b: u32) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (IndexSet::Medium((1u64 << 32) | (wrapped as u64)), true)
    } else {
        (IndexSet::Small(wrapped), false)
    }
}

#[inline]
fn add_medium(a: u64, b: u64) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (IndexSet::Large((1u128 << 64) | (wrapped as u128)), true)
    } else {
        (IndexSet::Medium(wrapped), false)
    }
}

#[inline]
fn add_large(a: u128, b: u128) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (
            IndexSet::Heap(vec![wrapped as u64, (wrapped >> 64) as u64, 1]),
            true,
        )
    } else {
        (IndexSet::Large(wrapped), false)
    }
}

#[inline]
fn add_heap(mut a: Vec<u64>, b: u64) -> (IndexSet, bool) {
    let mut carry = b;
    for item in a.iter_mut() {
        let (sum, ov) = item.overflowing_add(carry);
        *item = sum;
        carry = u64::from(ov);
        if carry == 0 {
            break;
        }
    }
    let overflow = carry > 0;
    if overflow {
        a.push(carry);
    }
    (IndexSet::Heap(a), overflow)
}

impl Sub for ArithIndexSet {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (result, underflow) = self.overflowing_sub(rhs);
        assert!(!underflow, "attempt to subtract with overflow");
        result
    }
}

impl SubAssign for ArithIndexSet {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) - rhs;
    }
}

impl ArithIndexSet {
    /// Integer subtraction with underflow. Returns `(diff, underflow)`
    ///
    /// `underflow` is `true` if `self < rhs` (result clamped to 0).
    #[must_use]
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (a, b) = (self.0, rhs.0);
        if a < b {
            return (ArithIndexSet(IndexSet::Small(0)), true);
        }
        let inner = match (a, b) {
            (IndexSet::Small(a), IndexSet::Small(b)) => IndexSet::Small(a - b),
            (IndexSet::Small(a), IndexSet::Medium(b)) => IndexSet::Medium(a as u64 - b),
            (IndexSet::Medium(a), IndexSet::Small(b)) => IndexSet::Medium(a - b as u64),
            (IndexSet::Medium(a), IndexSet::Medium(b)) => IndexSet::Medium(a - b),
            (IndexSet::Small(a), IndexSet::Large(b)) => IndexSet::Large(a as u128 - b),
            (IndexSet::Large(a), IndexSet::Small(b)) => IndexSet::Large(a - b as u128),
            (IndexSet::Medium(a), IndexSet::Large(b)) => IndexSet::Large(a as u128 - b),
            (IndexSet::Large(a), IndexSet::Medium(b)) => IndexSet::Large(a - b as u128),
            (IndexSet::Large(a), IndexSet::Large(b)) => IndexSet::Large(a - b),
            (IndexSet::Small(a), IndexSet::Heap(b)) => {
                IndexSet::Small(a - b.first().copied().unwrap_or(0) as u32)
            }
            (IndexSet::Heap(a), IndexSet::Small(b)) => sub_heap(a, b as u64),
            (IndexSet::Medium(a), IndexSet::Heap(b)) => {
                IndexSet::Medium(a - b.first().copied().unwrap_or(0))
            }
            (IndexSet::Heap(a), IndexSet::Medium(b)) => sub_heap(a, b),
            (a, b) => {
                let a: Vec<u64> = a.into();
                let b: Vec<u64> = b.into();
                let mut result = Vec::with_capacity(a.len());
                let mut borrow: bool = false;
                for i in 0..a.len() {
                    let av = a.get(i).copied().unwrap_or(0);
                    let bv = b.get(i).copied().unwrap_or(0);
                    let (diff, ov1) = av.overflowing_sub(bv);
                    let (diff2, ov2) = diff.overflowing_sub(u64::from(borrow));
                    result.push(diff2);
                    borrow = ov1 || ov2;
                }
                debug_assert!(
                    !borrow,
                    "subtraction underflow should have been caught earlier"
                );
                IndexSet::Heap(result)
            }
        }
        .minimized();
        (ArithIndexSet(inner), false)
    }

    /// Checked subtraction. Returns `None` if `self < rhs` (underflow).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (result, underflow) = self.overflowing_sub(rhs);
        if underflow { None } else { Some(result) }
    }
}

#[inline]
fn sub_heap(a: Vec<u64>, b: u64) -> IndexSet {
    let mut result = a;
    let mut borrow = b;
    for item in result.iter_mut() {
        let (diff, ov) = item.overflowing_sub(borrow);
        *item = diff;
        borrow = u64::from(ov);
        if borrow == 0 {
            break;
        }
    }
    debug_assert_eq!(
        borrow, 0,
        "subtraction underflow should have been caught earlier"
    );
    IndexSet::Heap(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singleton_and_contains() {
        let s = IndexSet::singleton(0);
        assert!(s.contains(0));
        assert!(!s.contains(1));

        let s = IndexSet::singleton(31);
        assert!(s.contains(31));

        let s = IndexSet::singleton(63);
        assert!(s.contains(63));

        let s = IndexSet::singleton(200);
        assert!(s.contains(200));
    }

    #[test]
    fn test_insert_and_growth() {
        let mut s = IndexSet::Small(0);
        s.insert(0, true);
        assert!(s.contains(0));

        s.insert(33, true);
        assert!(matches!(s, IndexSet::Medium(_)));
        assert!(s.contains(33));

        s.insert(70, true);
        assert!(matches!(s, IndexSet::Large(_)));
        assert!(s.contains(70));

        s.insert(150, true);
        assert!(matches!(s, IndexSet::Heap(_)));
        assert!(s.contains(150));

        s.insert(150, false);
        assert!(!s.contains(150));
    }

    #[test]
    fn test_iter() {
        let mut s = IndexSet::Small(0);
        s.insert(2, true);
        s.insert(5, true);
        s.insert(0, true);
        let v: Vec<_> = s.iter().collect();
        assert_eq!(v, vec![0, 2, 5]);
    }

    #[test]
    fn test_iter_large() {
        let mut s = IndexSet::singleton(200);
        s.insert(100, true);
        s.insert(0, true);
        let v: Vec<_> = s.iter().collect();
        assert_eq!(v, vec![0, 100, 200]);
    }

    #[test]
    fn test_iter_rev_and_exact() {
        let mut s = IndexSet::singleton(200);
        s.insert(100, true);
        s.insert(0, true);
        assert_eq!(s.iter().rev().collect::<Vec<_>>(), vec![200, 100, 0]);
        let mut it = s.iter();
        it.next();
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn test_iter_mixed() {
        let mut s = IndexSet::singleton(200);
        s.insert(5, true);
        s.insert(0, true);
        let mut it = s.iter();
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next_back(), Some(200));
        assert_eq!(it.next(), Some(5));
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_count_ones_and_is_empty() {
        assert!(IndexSet::Small(0).is_empty());
        assert_eq!(IndexSet::Small(0).count_ones(), 0);

        let mut s = IndexSet::Small(0);
        s.insert(0, true);
        s.insert(1, true);
        s.insert(2, true);
        assert_eq!(s.count_ones(), 3);
        assert!(!s.is_empty());

        let mut s = IndexSet::singleton(200);
        s.insert(201, true);
        assert_eq!(s.count_ones(), 2);
    }

    #[test]
    fn test_is_disjoint() {
        let a = IndexSet::singleton(0);
        let b = IndexSet::singleton(1);
        assert!(a.is_disjoint(&b));

        let b2 = IndexSet::singleton(0);
        assert!(!a.is_disjoint(&b2));
    }

    #[test]
    fn test_is_disjoint_cross_variant() {
        let s = IndexSet::singleton(0);
        let m = IndexSet::singleton(33);
        let l = IndexSet::singleton(70);
        let h = IndexSet::singleton(200);

        assert!(s.is_disjoint(&m));
        assert!(m.is_disjoint(&l));
        assert!(l.is_disjoint(&h));

        let overlap = IndexSet::singleton(0);
        assert!(!s.is_disjoint(&overlap));
    }

    #[test]
    fn test_minimize() {
        // Large with value that fits in Small
        let mut s = IndexSet::Large(0);
        s.insert(5, true);
        let s = s.minimized();
        assert!(matches!(s, IndexSet::Small(_)));

        // Heap that can downgrade to Large
        let mut s = IndexSet::Heap(vec![0, 0]);
        s.insert(70, true);
        let s = s.minimized();
        assert!(matches!(s, IndexSet::Large(_)));

        // Already minimal
        let s = IndexSet::singleton(0).minimized();
        assert!(matches!(s, IndexSet::Small(_)));
    }

    #[test]
    fn test_bitor() {
        let a = IndexSet::singleton(0);
        let b = IndexSet::singleton(1);
        let c = a | b;
        assert!(c.contains(0));
        assert!(c.contains(1));

        let v: Vec<_> = c.iter().collect();
        assert_eq!(v, vec![0, 1]);
    }

    #[test]
    fn test_bitand() {
        let a = IndexSet::singleton(0) | IndexSet::singleton(1);
        let b = IndexSet::singleton(1) | IndexSet::singleton(2);
        let c = a & b;
        assert!(!c.contains(0));
        assert!(c.contains(1));
        assert!(!c.contains(2));
    }

    #[test]
    fn test_partial_eq_across_variants() {
        let s = IndexSet::singleton(0);
        let m = IndexSet::Medium(1);
        assert_eq!(s, m);

        let m2 = IndexSet::Medium(1);
        assert_eq!(s, m2);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let a = IndexSet::singleton(0) | IndexSet::singleton(1);
        let b = IndexSet::Medium(0b11);
        let ha = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hb = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(ha, hb);
    }

    #[test]
    fn test_default_is_empty() {
        let s: IndexSet = Default::default();
        assert!(s.is_empty());
        assert!(matches!(s, IndexSet::Small(0)));
    }

    #[test]
    fn test_shl_small() {
        let s = IndexSet::singleton(0) | IndexSet::singleton(2);
        let shifted = s << 3;
        assert!(shifted.contains(3));
        assert!(shifted.contains(5));
        assert!(!shifted.contains(0));
    }

    #[test]
    fn test_shl_overflow_small() {
        let s = IndexSet::singleton(0) | IndexSet::singleton(31);
        let shifted = s << 1;
        assert!(shifted.contains(1));
        assert!(shifted.contains(32));
    }

    #[test]
    fn test_shl_large_to_heap() {
        let s = IndexSet::singleton(120);
        let shifted = s << 10;
        assert!(shifted.contains(130));
    }

    #[test]
    fn test_shr_small() {
        let s = IndexSet::singleton(5) | IndexSet::singleton(8);
        let shifted = s >> 3;
        assert!(shifted.contains(2));
        assert!(shifted.contains(5));
        assert!(!shifted.contains(0));
    }

    #[test]
    fn test_shl_shr_empty() {
        let s = IndexSet::Small(0);
        assert!((s.clone() << 5).is_empty());
        assert!((s >> 5).is_empty());
    }

    #[test]
    fn test_shl_zero_shift() {
        let s = IndexSet::singleton(0) | IndexSet::singleton(5) | IndexSet::singleton(100);
        assert_eq!(s.clone() << 0, s);
    }

    #[test]
    fn test_shr_overflow() {
        let s = IndexSet::singleton(0)
            | IndexSet::singleton(33)
            | IndexSet::singleton(70)
            | IndexSet::singleton(200);
        assert!((s >> 300).is_empty());
    }

    #[test]
    fn test_shl_shr_roundtrip() {
        let s = IndexSet::singleton(0) | IndexSet::singleton(5) | IndexSet::singleton(100);
        let shifted = s.clone() << 7 >> 7;
        assert_eq!(shifted, s);
    }

    #[test]
    fn test_ord_same_variant() {
        assert!(IndexSet::Small(0b001) < IndexSet::Small(0b010));
        assert!(IndexSet::Small(0b010) > IndexSet::Small(0b001));
        assert_eq!(IndexSet::Small(0b101), IndexSet::Small(0b101));
        assert!(IndexSet::Medium(0b001) < IndexSet::Medium(0b010));
        assert!(IndexSet::Large(0b001) < IndexSet::Large(0b010));
    }

    #[test]
    fn test_ord_cross_variant() {
        assert_eq!(IndexSet::Small(0b11), IndexSet::Medium(0b11));
        assert_eq!(IndexSet::Medium(0b11), IndexSet::Large(0b11));
        assert_eq!(IndexSet::Large(0b11), IndexSet::Heap(vec![0b11]));
        assert!(IndexSet::Small(0b100) > IndexSet::Large(0b010));
    }

    #[test]
    fn test_arith_add_small() {
        let a = ArithIndexSet(IndexSet::Small(5));
        let b = ArithIndexSet(IndexSet::Small(3));
        let c = a + b;
        assert_eq!(c.0, IndexSet::Small(8));
    }

    #[test]
    fn test_arith_add_with_carry() {
        let a = ArithIndexSet(IndexSet::Small(u32::MAX));
        let b = ArithIndexSet(IndexSet::Small(1));
        let c = a + b;
        assert!(!matches!(c.0, IndexSet::Small(_)));
        assert_eq!(c.0.count_ones(), 1);
        assert!(c.0.contains(32));
    }

    #[test]
    fn test_arith_add_large_overflow() {
        let a = ArithIndexSet(IndexSet::Large(u128::MAX));
        let b = ArithIndexSet(IndexSet::Large(1));
        let c = a + b;
        if let IndexSet::Heap(v) = &c.0 {
            assert_eq!(v.len(), 3);
            assert_eq!(v[2], 1);
        } else {
            panic!("expected Heap variant");
        }
    }

    #[test]
    fn test_arith_sub_small() {
        let a = ArithIndexSet(IndexSet::Small(8));
        let b = ArithIndexSet(IndexSet::Small(3));
        let c = a - b;
        assert_eq!(c.0, IndexSet::Small(5));
    }

    #[test]
    fn test_arith_sub_underflow_panics() {
        let a = ArithIndexSet(IndexSet::Small(3));
        let b = ArithIndexSet(IndexSet::Small(8));
        assert!(std::panic::catch_unwind(move || a - b).is_err());
    }

    #[test]
    fn test_arith_from_conversion() {
        let s = IndexSet::Small(42);
        let a: ArithIndexSet = s.into();
        let back: IndexSet = a.into();
        assert_eq!(back, IndexSet::Small(42));
    }

    #[test]
    fn test_arith_deref() {
        let a = ArithIndexSet(IndexSet::Small(0b101));
        assert!(a.contains(0));
        assert!(a.contains(2));
    }

    #[test]
    fn test_arith_add_heap_fallback() {
        let a = ArithIndexSet(IndexSet::Heap(vec![u64::MAX, u64::MAX, 1]));
        let b = ArithIndexSet(IndexSet::Heap(vec![1, 0, 1]));
        let c = a + b;
        if let IndexSet::Heap(v) = c.0 {
            assert_eq!(v, vec![0, 0, 3]);
        } else {
            panic!("expected Heap variant");
        }
    }

    #[test]
    fn test_bitor_assign() {
        let mut a = IndexSet::singleton(0);
        let b = IndexSet::singleton(1);
        a |= b;
        assert!(a.contains(0));
        assert!(a.contains(1));
    }

    #[test]
    fn test_arith_assign() {
        let mut a = ArithIndexSet(IndexSet::Small(5));
        a += ArithIndexSet(IndexSet::Small(3));
        assert_eq!(a.0, IndexSet::Small(8));
        a -= ArithIndexSet(IndexSet::Small(3));
        assert_eq!(a.0, IndexSet::Small(5));
    }

    #[test]
    fn test_arith_sub_heap_fallback() {
        let a = ArithIndexSet(IndexSet::Heap(vec![2, 1, 1]));
        let b = ArithIndexSet(IndexSet::Heap(vec![1, 1, 0]));
        let c = a - b;
        if let IndexSet::Heap(v) = c.0 {
            assert_eq!(v, vec![1, 0, 1]);
        } else {
            panic!("expected Heap variant");
        }
    }
}
