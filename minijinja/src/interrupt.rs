//! Cooperative interruption of template evaluation.
//!
//! MiniJinja charges fuel per VM instruction only.  Native work such
//! as `x in seq`, sequence equality or printing a lazily repeated sequence
//! (`range(100000) * 10**12`) iterates inside a single instruction, so neither
//! fuel nor a caller's deadline can stop it.  A caller that needs a hard
//! deadline installs a flag for the rendering thread with
//! [`set_thread_interrupt`] and sets it when the deadline passes.  From then
//! on every native object iteration on that thread ends at its next element,
//! and the VM fails at its next instruction, so the render returns an error
//! promptly instead of finishing the native loop.
//!
//! Threads that never install a flag behave exactly as before: iterators are
//! not wrapped and the VM performs one extra `None` check per instruction.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::{Error, ErrorKind};

thread_local! {
    static INTERRUPT: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Installs (or with `None` removes) the interrupt flag for the current thread
/// and returns the previously installed flag.
///
/// While a flag is installed and set, object iteration started on this thread
/// yields no further elements and template evaluation on this thread fails
/// with [`ErrorKind::InvalidOperation`] at its next instruction.  Results
/// computed after the flag is set are therefore meaningless; callers must
/// discard them, which a deadline-bounded caller does anyway.
pub fn set_thread_interrupt(flag: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    INTERRUPT.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), flag))
}

/// The interrupt flag installed for the current thread, if any.
pub(crate) fn current() -> Option<Arc<AtomicBool>> {
    INTERRUPT.with(|slot| slot.borrow().clone())
}

/// Whether the current thread's interrupt flag is installed and set.
pub(crate) fn is_interrupted() -> bool {
    INTERRUPT.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    })
}

/// The error evaluation fails with once the flag is set.
pub(crate) fn interrupted_error() -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        "template evaluation was interrupted",
    )
}

/// An iterator that ends as soon as `flag` is set.
struct Interruptible<I> {
    flag: Arc<AtomicBool>,
    inner: I,
}

impl<I: Iterator> Iterator for Interruptible<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.flag.load(Ordering::Relaxed) {
            None
        } else {
            self.inner.next()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Until the flag is set the iterator yields exactly what the inner one
        // does, so its hint is reported unchanged: composed lazy wrappers
        // (`range(4)[1:]`, `items`), `|length` and loop metadata stay exactly
        // unchanged. Once set, iteration ends at the next element, so the
        // only true lower bound is zero.
        let (lower, upper) = self.inner.size_hint();
        if self.flag.load(Ordering::Relaxed) {
            (0, upper)
        } else {
            (lower, upper)
        }
    }
}

/// Largest capacity an interrupt-enabled thread reserves for sequence
/// concatenation from a known (possibly lazy, virtual) length: 2^20 values,
/// about 24 MiB.
pub(crate) const PREALLOC_CEILING: usize = 1 << 20;

/// Collects an iterator into a `Vec`. Without an interrupt installed this is
/// a plain `collect()`. With one installed it reserves at most
/// [`PREALLOC_CEILING`] from the size hint and pushes element by element, so a
/// lazy sequence's exact virtual length (10^17 for `range(100000) * 10**12`) is
/// never reserved up front and the walk can reach the interrupt. Every native
/// MiniJinja collector of template values goes through this or [`extend_vec`].
pub(crate) fn collect_vec<T, I: Iterator<Item = T>>(iter: I) -> Vec<T> {
    let mut rv = Vec::new();
    extend_vec(&mut rv, iter);
    rv
}

/// `Vec::extend` with the same bounded reservation as [`collect_vec`].
pub(crate) fn extend_vec<T, I: Iterator<Item = T>>(rv: &mut Vec<T>, iter: I) {
    if current().is_some() {
        rv.reserve(iter.size_hint().0.min(PREALLOC_CEILING));
        for item in iter {
            rv.push(item);
        }
    } else {
        rv.extend(iter);
    }
}

/// The capacity to reserve for `len` elements: `len` unchanged on threads with
/// no interrupt installed, capped at [`PREALLOC_CEILING`] otherwise.
pub(crate) fn prealloc(len: usize) -> usize {
    if current().is_some() {
        len.min(PREALLOC_CEILING)
    } else {
        len
    }
}

/// A non-boxed iterator that ends once the flag captured at creation is set.
/// Used for native string and byte walks; with no flag installed it only
/// forwards to the inner iterator.
pub(crate) struct Guarded<I> {
    flag: Option<Arc<AtomicBool>>,
    inner: I,
}

impl<I: Iterator> Iterator for Guarded<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.flag {
            Some(ref flag) if flag.load(Ordering::Relaxed) => None,
            _ => self.inner.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.flag {
            // Ends at the next element once interrupted.
            Some(ref flag) if flag.load(Ordering::Relaxed) => (0, self.inner.size_hint().1),
            _ => self.inner.size_hint(),
        }
    }
}

/// Bytes of a heap string counted between two checks of the interrupt flag.
const COUNT_CHUNK_BYTES: usize = 1 << 16;

/// Counts the characters of a string of up to 100 MB. With an interrupt
/// installed the count checks the flag every [`COUNT_CHUNK_BYTES`] bytes and
/// stops once it is set, returning a partial count the caller's failing render
/// discards; without one it is a plain `chars().count()`.
pub(crate) fn char_count(s: &str) -> usize {
    match current() {
        Some(flag) => count_chars_until(s, COUNT_CHUNK_BYTES, || flag.load(Ordering::Relaxed)),
        None => s.chars().count(),
    }
}

/// Counts characters in `chunk`-byte steps (widened to a character boundary),
/// asking `stop` before each step and returning the count so far once it
/// answers true.
fn count_chars_until(s: &str, chunk: usize, mut stop: impl FnMut() -> bool) -> usize {
    let mut count = 0;
    let mut rest = s;
    while !rest.is_empty() && !stop() {
        let mut end = chunk.min(rest.len());
        while !rest.is_char_boundary(end) {
            end += 1;
        }
        count += rest[..end].chars().count();
        rest = &rest[end..];
    }
    count
}

/// Wraps a native string or byte walk so it ends once the current thread's
/// interrupt flag is set.
pub(crate) fn guard_native<I: Iterator>(inner: I) -> Guarded<I> {
    Guarded {
        flag: current(),
        inner,
    }
}

/// Wraps a native object iterator so it ends once the current thread's
/// interrupt flag is set.  Without an installed flag the iterator is returned
/// unchanged.
pub(crate) fn guard<'a, T: 'a>(
    iter: Box<dyn Iterator<Item = T> + Send + Sync + 'a>,
) -> Box<dyn Iterator<Item = T> + Send + Sync + 'a> {
    match current() {
        Some(flag) => Box::new(Interruptible { flag, inner: iter }),
        None => iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::value::ops;
    use crate::value::Value;

    /// Runs `body` with a set interrupt installed, then restores the slot.
    fn while_interrupted<T>(body: impl FnOnce() -> T) -> T {
        let previous = set_thread_interrupt(Some(Arc::new(AtomicBool::new(true))));
        let result = body();
        set_thread_interrupt(previous);
        result
    }

    #[test]
    fn guarded_hint_is_unchanged_until_the_flag_is_set_then_zero() {
        let huge = 10usize.pow(17);
        let flag = Arc::new(AtomicBool::new(false));
        let previous = set_thread_interrupt(Some(flag.clone()));
        let iter = guard(Box::new((0..huge).map(Value::from)));
        assert_eq!(iter.size_hint(), (huge, Some(huge)), "unset: inner hint");
        let lazy = ops::mul(
            &Value::from(vec![Value::from(0); 100_000]),
            &Value::from(10u64.pow(12)),
        )
        .unwrap();
        let value_iter = lazy.try_iter().unwrap();
        assert_eq!(value_iter.size_hint(), (huge, Some(huge)));
        assert_eq!(value_iter.exact_len(), Some(huge));
        let text = guard_native("abcdef".chars());
        assert_eq!(text.size_hint(), "abcdef".chars().size_hint());
        flag.store(true, Ordering::Relaxed);
        assert_eq!(iter.size_hint(), (0, Some(huge)), "set: zero lower bound");
        assert_eq!(value_iter.size_hint(), (0, Some(huge)));
        assert_eq!(text.size_hint(), (0, Some(6)));
        // With a zero lower bound the collector reserves nothing up front,
        // reaches `next` and stops.
        let collected: Vec<Value> = iter.collect();
        assert!(collected.is_empty());
        assert_eq!(value_iter.count(), 0);
        set_thread_interrupt(previous);
    }

    /// A sequence that sets the installed interrupt flag when element
    /// `trip_at` is read, counting every read.
    #[derive(Debug)]
    struct TripAt {
        trip_at: usize,
        flag: Arc<AtomicBool>,
        reads: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::value::Object for TripAt {
        fn repr(self: &Arc<Self>) -> crate::value::ObjectRepr {
            crate::value::ObjectRepr::Seq
        }

        fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
            let index = key.as_usize().filter(|index| *index < 1_000)?;
            let reads = self.reads.fetch_add(1, Ordering::Relaxed) + 1;
            assert!(reads <= 1_000_000, "the interrupt was ignored");
            if index == self.trip_at {
                self.flag.store(true, Ordering::Relaxed);
            }
            Some(Value::from(index))
        }

        fn enumerate(self: &Arc<Self>) -> crate::value::Enumerator {
            crate::value::Enumerator::Seq(1_000)
        }
    }

    #[test]
    fn backward_slice_collects_a_lazy_sequence_without_reserving_it() {
        // The collection starts with the flag unset, so its source reports
        // the exact virtual length 10^15; reserving that would abort.
        let flag = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let previous = set_thread_interrupt(Some(flag.clone()));
        let seq = Value::from_object(TripAt {
            trip_at: 499,
            flag: flag.clone(),
            reads: reads.clone(),
        });
        let lazy = ops::mul(&seq, &Value::from(10u64.pow(12))).unwrap();
        let reversed = ops::slice(lazy, Value::from(()), Value::from(()), Value::from(-1)).unwrap();
        let _ = reversed.try_iter().unwrap().count();
        assert_eq!(reads.load(Ordering::Relaxed), 500);
        set_thread_interrupt(previous);
    }

    #[test]
    fn deep_concatenation_materializes_a_lazy_sequence_without_reserving_it() {
        // Past 32 nested `+` the chain is materialized from its virtual length
        // (here 10^15 + 33). Reserving that length would abort before
        // reading anything; the walk must instead reach the interrupt.
        let flag = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let previous = set_thread_interrupt(Some(flag.clone()));
        let seq = Value::from_object(TripAt {
            trip_at: 499,
            flag: flag.clone(),
            reads: reads.clone(),
        });
        let mut chain = ops::mul(&seq, &Value::from(10u64.pow(12))).unwrap();
        for _ in 0..33 {
            chain = ops::add(&chain, &Value::from(vec![Value::from(1)])).unwrap();
        }
        assert_eq!(reads.load(Ordering::Relaxed), 500);
        assert!(flag.load(Ordering::Relaxed));
        set_thread_interrupt(previous);
    }

    #[test]
    fn collectors_reserve_at_most_the_ceiling_when_installed() {
        let huge = 10usize.pow(17);
        // Without an interrupt this is `collect()` (here over a short range).
        assert_eq!(collect_vec(0..4), vec![0, 1, 2, 3]);
        let previous = set_thread_interrupt(Some(Arc::new(AtomicBool::new(false))));
        // An iterator claiming 10^17 exact elements but yielding three: the
        // collector reserves at most the ceiling, never the claimed length.
        struct Claims(std::ops::Range<usize>, usize);
        impl Iterator for Claims {
            type Item = usize;
            fn next(&mut self) -> Option<usize> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.1, Some(self.1))
            }
        }
        let collected = collect_vec(Claims(0..3, huge));
        assert_eq!(collected, vec![0, 1, 2]);
        assert!(collected.capacity() <= PREALLOC_CEILING);
        let mut extended = vec![9];
        extend_vec(&mut extended, Claims(0..2, huge));
        assert_eq!(extended, vec![9, 0, 1]);
        assert!(extended.capacity() <= PREALLOC_CEILING + 1);
        set_thread_interrupt(previous);
    }

    #[test]
    fn counting_stops_between_chunks_once_asked() {
        let text = "aé€😀".repeat(10_000); // 1-, 2-, 3- and 4-byte characters
        assert_eq!(count_chars_until(&text, 7, || false), text.chars().count());
        let mut checks = 0;
        let partial = count_chars_until(&text, 1024, || {
            checks += 1;
            checks > 3
        });
        // Exactly three chunks (each widened to a character boundary) were
        // counted before the stop; the rest of the string was never scanned.
        let three_chunks = {
            let mut rest = text.as_str();
            let mut count = 0;
            for _ in 0..3 {
                let mut end = 1024;
                while !rest.is_char_boundary(end) {
                    end += 1;
                }
                count += rest[..end].chars().count();
                rest = &rest[end..];
            }
            count
        };
        assert_eq!(partial, three_chunks);
        assert!(partial < text.chars().count());
    }

    #[test]
    fn heap_string_counts_stop_once_interrupted() {
        let text = Value::from("abcdef".repeat(100_000));
        assert_eq!(text.len(), Some(600_000));
        while_interrupted(|| {
            assert_eq!(text.len(), Some(0));
            assert_eq!(char_count(text.as_str().unwrap()), 0);
        });
    }

    /// `value[1:]` for a positive `step`, `value[::step]` otherwise.
    fn sliced(value: &Value, step: i64) -> Value {
        let start = if step > 0 {
            Value::from(1)
        } else {
            Value::from(())
        };
        ops::slice(value.clone(), start, Value::from(()), Value::from(step)).unwrap()
    }

    #[test]
    fn string_and_byte_slicing_stop_once_interrupted() {
        let text = Value::from("abcdef");
        let bytes = Value::from_bytes(b"abcdef".to_vec());
        assert_eq!(sliced(&text, 1), Value::from("bcdef"));
        assert_eq!(sliced(&text, -1), Value::from("fedcba"));
        assert_eq!(sliced(&bytes, 1), Value::from_bytes(b"bcdef".to_vec()));
        assert_eq!(sliced(&bytes, -1), Value::from_bytes(b"fedcba".to_vec()));
        while_interrupted(|| {
            assert_eq!(sliced(&text, 1), Value::from(""));
            assert_eq!(sliced(&text, -1), Value::from(""));
            assert_eq!(sliced(&bytes, 1), Value::from_bytes(Vec::new()));
            assert_eq!(sliced(&bytes, -1), Value::from_bytes(Vec::new()));
        });
    }

    #[test]
    fn string_and_byte_reversal_stop_once_interrupted() {
        // Long enough to be a heap string: inline small strings (a few bytes)
        // keep the unguarded reversal.
        let text = Value::from("abcdef".repeat(8));
        let bytes = Value::from_bytes(b"abcdef".to_vec());
        assert_eq!(text.reverse().unwrap(), Value::from("fedcba".repeat(8)));
        while_interrupted(|| {
            assert_eq!(text.reverse().unwrap(), Value::from(""));
            assert_eq!(bytes.reverse().unwrap(), Value::from_bytes(Vec::new()));
            assert_eq!(text.try_iter().unwrap().count(), 0);
        });
    }

    #[test]
    fn concatenation_reserves_at_most_the_ceiling_when_installed() {
        assert_eq!(prealloc(usize::MAX), usize::MAX);
        let previous = set_thread_interrupt(Some(Arc::new(AtomicBool::new(false))));
        assert_eq!(prealloc(usize::MAX), PREALLOC_CEILING);
        assert_eq!(prealloc(7), 7);
        set_thread_interrupt(previous);
    }
}
