use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use smallvec::SmallVec;

use crate::{NpError, NpResult};

const INLINE_PARITY_TARGETS: usize = 8;

pub(crate) fn canonical_id_order<'a>(
    declared_ids: &[i64],
    edge_targets: impl IntoIterator<Item = &'a [i64]>,
    duplicate_declaration_message: &'static str,
) -> NpResult<Vec<i64>> {
    let mut ids = Vec::with_capacity(declared_ids.len());
    let mut seen = HashSet::with_capacity(declared_ids.len());
    for id in declared_ids {
        if !seen.insert(*id) {
            return Err(NpError::new(duplicate_declaration_message));
        }
        ids.push(*id);
    }
    for targets in edge_targets {
        for id in targets {
            if seen.insert(*id) {
                ids.push(*id);
            }
        }
    }
    Ok(ids)
}

pub(crate) fn parity_canonicalize<T>(targets: &mut Vec<T>)
where
    T: Copy + Eq + Hash,
{
    match targets.len() {
        0 | 1 => return,
        2 => {
            if targets[0] == targets[1] {
                targets.clear();
            }
            return;
        }
        _ => {}
    }

    if targets.len() <= INLINE_PARITY_TARGETS {
        let states = inline_parity_states(targets);
        targets.clear();
        targets.extend(
            states
                .into_iter()
                .filter_map(|(target, odd)| odd.then_some(target)),
        );
        return;
    }

    let mut states: HashMap<T, (usize, bool)> = HashMap::with_capacity(targets.len());
    for (index, target) in targets.iter().copied().enumerate() {
        match states.entry(target) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                state.1 = !state.1;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((index, true));
            }
        }
    }

    let mut index = 0;
    targets.retain(|target| {
        let (first_index, odd) = states
            .get(target)
            .expect("every target must have a parity state");
        let keep = *first_index == index && *odd;
        index += 1;
        keep
    });
}

pub(crate) fn parity_support_len<T>(targets: &[T]) -> usize
where
    T: Copy + Eq + Hash,
{
    match targets.len() {
        0 => return 0,
        1 => return 1,
        2 => return usize::from(targets[0] != targets[1]) * 2,
        _ => {}
    }

    if targets.len() <= INLINE_PARITY_TARGETS {
        return inline_parity_states(targets)
            .iter()
            .filter(|(_, odd)| *odd)
            .count();
    }

    let mut odd = HashSet::with_capacity(targets.len());
    for target in targets {
        if !odd.insert(*target) {
            odd.remove(target);
        }
    }
    odd.len()
}

fn inline_parity_states<T>(targets: &[T]) -> SmallVec<[(T, bool); INLINE_PARITY_TARGETS]>
where
    T: Copy + Eq,
{
    debug_assert!(targets.len() <= INLINE_PARITY_TARGETS);
    let mut states: SmallVec<[(T, bool); INLINE_PARITY_TARGETS]> = SmallVec::new();
    for target in targets.iter().copied() {
        if let Some(state) = states.iter_mut().find(|state| state.0 == target) {
            state.1 = !state.1;
        } else {
            states.push((target, true));
        }
    }
    states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_id_order_preserves_declarations_then_raw_edge_discovery() {
        let first = [9, 9, 3];
        let second = [7, 2, 7];

        let ids = canonical_id_order(
            &[5, 2],
            [first.as_slice(), second.as_slice()],
            "duplicate ids",
        )
        .unwrap();

        assert_eq!(ids, vec![5, 2, 9, 3, 7]);
    }

    #[test]
    fn canonical_id_order_rejects_duplicate_declarations() {
        let err = canonical_id_order(&[5, 5], [], "duplicate ids").unwrap_err();

        assert_eq!(err.message(), "duplicate ids");
    }

    #[test]
    fn parity_canonicalization_keeps_odd_targets_in_first_occurrence_order() {
        let mut interleaved = vec![4, 2, 4, 3, 2, 4, 1, 1];
        parity_canonicalize(&mut interleaved);
        assert_eq!(interleaved, vec![4, 3]);
        assert_eq!(parity_support_len(&[4, 2, 4, 3, 2, 4, 1, 1]), 2);

        let mut cancelled = vec![9, 9];
        parity_canonicalize(&mut cancelled);
        assert!(cancelled.is_empty());
        assert_eq!(parity_support_len(&[9, 9]), 0);
    }

    #[test]
    fn parity_canonicalization_preserves_order_on_large_supports() {
        let mut targets = vec![9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2];

        parity_canonicalize(&mut targets);

        assert_eq!(targets, vec![1, 3, 4, 5, 6, 7, 8]);
        assert_eq!(parity_support_len(&[9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2]), 7);
    }

    #[test]
    fn parity_fast_paths_match_reference_for_small_alphabet() {
        for len in 0u32..=10 {
            for mut encoded in 0..3usize.pow(len) {
                let mut raw = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    raw.push((encoded % 3) as i64);
                    encoded /= 3;
                }
                let expected = raw
                    .iter()
                    .enumerate()
                    .filter_map(|(index, target)| {
                        (!raw[..index].contains(target)
                            && raw
                                .iter()
                                .filter(|candidate| **candidate == *target)
                                .count()
                                % 2
                                == 1)
                            .then_some(*target)
                    })
                    .collect::<Vec<_>>();

                let mut actual = raw.clone();
                parity_canonicalize(&mut actual);

                assert_eq!(actual, expected, "raw targets: {raw:?}");
                assert_eq!(parity_support_len(&raw), expected.len());
            }
        }
    }
}
