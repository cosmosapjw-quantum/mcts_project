use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static HOT_PATH_METRICS_OVERRIDE: AtomicU8 = AtomicU8::new(0);
static HOT_PATH_METRICS_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn hot_path_metrics_enabled() -> bool {
    match HOT_PATH_METRICS_OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => *HOT_PATH_METRICS_ENABLED.get_or_init(|| {
            std::env::var("QUARTZ_MCTS_HOTPATH_METRICS")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        }),
    }
}

#[inline]
pub fn maybe_start_timer() -> Option<Instant> {
    hot_path_metrics_enabled().then(Instant::now)
}

#[inline]
pub fn record_elapsed_nanos(counter: &AtomicU64, started: Option<Instant>) {
    if let Some(t0) = started {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
pub fn set_test_hot_path_metrics_override(enabled: Option<bool>) {
    let raw = match enabled {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    HOT_PATH_METRICS_OVERRIDE.store(raw, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    use std::time::Duration;

    struct MetricsOverrideGuard;

    impl MetricsOverrideGuard {
        fn new(enabled: bool) -> Self {
            set_test_hot_path_metrics_override(Some(enabled));
            MetricsOverrideGuard
        }
    }

    impl Drop for MetricsOverrideGuard {
        fn drop(&mut self) {
            set_test_hot_path_metrics_override(None);
        }
    }

    #[test]
    fn test_hot_path_metrics_override_disable_blocks_timer_records() {
        let _guard = MetricsOverrideGuard::new(false);
        let counter = AtomicU64::new(0);
        assert!(!hot_path_metrics_enabled());
        let started = maybe_start_timer();
        assert!(started.is_none());
        record_elapsed_nanos(&counter, started);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hot_path_metrics_override_enable_records_timer() {
        let _guard = MetricsOverrideGuard::new(true);
        let counter = AtomicU64::new(0);
        assert!(hot_path_metrics_enabled());
        let started = maybe_start_timer();
        assert!(started.is_some());
        thread::sleep(Duration::from_micros(50));
        record_elapsed_nanos(&counter, started);
        assert!(counter.load(Ordering::Relaxed) > 0);
    }
}
