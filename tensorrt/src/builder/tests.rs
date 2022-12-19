use super::*;
use crate::dims::Dims3;
use crate::network::Network;
use crate::uff::{UffFile, UffInputOrder, UffParser};
use lazy_static::lazy_static;
use std::path::Path;
use std::sync::Mutex;

lazy_static! {
    static ref LOGGER: Mutex<Logger> = Mutex::new(Logger::new());
}

fn create_network(logger: &Logger) -> (Network, Builder) {
    let builder = Builder::new(&logger);
    let network = builder.create_network_v2(NetworkBuildFlags::DEFAULT);

    let uff_parser = UffParser::new();
    let dim = Dims3::new(1, 28, 28);

    uff_parser
        .register_input("in", dim, UffInputOrder::Nchw)
        .unwrap();
    uff_parser.register_output("out").unwrap();
    println!(
        "current dir: {}",
        std::env::current_dir().unwrap().display()
    );
    let uff_file = UffFile::new(Path::new("../assets/lenet5.uff")).unwrap();
    uff_parser.parse(&uff_file, &network).unwrap();

    (network, builder)
}

#[test]
fn platform_has_fast_fp16() {
    let logger = match LOGGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let builder = Builder::new(&logger);

    assert_eq!(builder.platform_has_fast_fp16(), true);
}

#[test]
fn platform_has_fast_int8() {
    let logger = match LOGGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let builder = Builder::new(&logger);

    assert_eq!(builder.platform_has_fast_int8(), true);
}

#[test]
fn get_max_dla_batch_size() {
    let logger = match LOGGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let builder = Builder::new(&logger);

    assert_eq!(builder.get_max_dla_batch_size(), 1);
}

#[test]
fn get_nb_dla_cores() {
    let logger = match LOGGER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let builder = Builder::new(&logger);

    assert_eq!(builder.get_nb_dla_cores(), 0);
}
