//! Serve exit must join the model thread (GitHub: Linux VRAM zombie).
//!
//! Dropping [`std::thread::JoinHandle`] without [`JoinHandle::join`] detaches
//! `ignis-model`. On a CUDA load that thread can hang in driver teardown and
//! leave the process a zombie still holding VRAM. `Server::serve_on_until`
//! therefore calls [`ignis_server::ActiveModel::shut_down`] after the HTTP
//! drain — this test proves the driver handle is gone once serve returns.

use std::sync::Arc;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, Server};

#[tokio::test]
async fn serve_on_until_joins_the_model_thread_before_returning() {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "qwen3.8-flash-next".into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let (engine, driver) =
        Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let model = ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver);
    assert!(
        !model.driver_joined(),
        "a model built with_driver keeps its JoinHandle until shut_down"
    );

    let server = Server::from_active(model);
    let active = server.active();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    // Resolve immediately: we only care about the post-drain join.
    server
        .serve_on_until(listener, None, std::future::ready(()))
        .await
        .expect("serve");

    assert!(
        active.driver_joined(),
        "serve_on_until must shut_down/join the model thread before returning"
    );
}
