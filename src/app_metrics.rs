use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;

// Define Metrics trait locally since ai-lib-rust doesn't have it
#[async_trait]
#[allow(dead_code)] // 部分方法为扩展预留，当前示例未全部使用
pub trait Metrics: Send + Sync {
    async fn incr_counter(&self, name: &str, value: u64);
    async fn record_gauge(&self, name: &str, value: f64);
    async fn start_timer(&self, name: &str) -> Option<Box<dyn Timer + Send>>;
    async fn record_histogram(&self, name: &str, value: f64);
    async fn record_histogram_with_tags(&self, name: &str, value: f64, tags: &[(&str, &str)]);
    async fn incr_counter_with_tags(&self, name: &str, value: u64, tags: &[(&str, &str)]);
    async fn record_gauge_with_tags(&self, name: &str, value: f64, tags: &[(&str, &str)]);
    async fn record_error(&self, name: &str, error_type: &str);
    async fn record_success(&self, name: &str, success: bool);
}

pub trait Timer: Send {
    fn stop(self: Box<Self>);
}

pub struct SimpleMetrics;

pub struct SimpleTimer {
    name: String,
    start: Instant,
}

impl SimpleMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Timer for SimpleTimer {
    fn stop(self: Box<Self>) {
        let elapsed = self.start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        println!("[metrics] timer {}: {:.2} ms", self.name, ms);
    }
}

#[async_trait]
impl Metrics for SimpleMetrics {
    async fn incr_counter(&self, name: &str, value: u64) {
        println!("[metrics] counter {} += {}", name, value);
    }

    async fn record_gauge(&self, name: &str, value: f64) {
        println!("[metrics] gauge {} = {:.3}", name, value);
    }

    async fn start_timer(&self, name: &str) -> Option<Box<dyn Timer + Send>> {
        Some(Box::new(SimpleTimer {
            name: name.to_string(),
            start: Instant::now(),
        }))
    }

    async fn record_histogram(&self, name: &str, value: f64) {
        println!("[metrics] hist {} = {:.3}", name, value);
    }

    async fn record_histogram_with_tags(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        println!("[metrics] hist {} = {:.3} tags={:?}", name, value, tags);
    }

    async fn incr_counter_with_tags(&self, name: &str, value: u64, tags: &[(&str, &str)]) {
        println!("[metrics] counter {} += {} tags={:?}", name, value, tags);
    }

    async fn record_gauge_with_tags(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        println!("[metrics] gauge {} = {:.3} tags={:?}", name, value, tags);
    }

    async fn record_error(&self, name: &str, error_type: &str) {
        println!("[metrics] error {} type={}", name, error_type);
    }

    async fn record_success(&self, name: &str, success: bool) {
        println!("[metrics] success {} = {}", name, success);
    }
}
