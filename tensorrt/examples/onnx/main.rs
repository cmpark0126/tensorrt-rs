use ndarray::Array;
use ndarray_image;
use std::iter::FromIterator;
use std::path::PathBuf;
use tensorrt_rs::builder::{Builder, NetworkBuildFlags};
use tensorrt_rs::context::ExecuteInput;
use tensorrt_rs::data_size::GB;
use tensorrt_rs::dims::{Dims2, Dims4};
use tensorrt_rs::engine::Engine;
use tensorrt_rs::onnx::{OnnxFile, OnnxParser};
use tensorrt_rs::runtime::{Logger, Runtime};

fn create_engine(
    logger: &Logger,
    file: OnnxFile,
    batch_sizes: Vec<i32>,
    workspace_size: usize,
) -> Engine {
    let max_batch_size = *batch_sizes.iter().max().unwrap();

    let builder = Builder::new(logger);
    builder.set_max_batch_size(max_batch_size);

    let network = builder.create_network_v2(NetworkBuildFlags::EXPLICIT_BATCH);
    let verbosity = 7;

    let parser = OnnxParser::new(&network, logger);
    parser.parse_from_file(&file, verbosity).unwrap();
    drop(parser);

    let dim = Dims4::new(max_batch_size, 3, 224, 224);
    network.get_input(0).set_dimensions(dim);
    let input_name = network.get_input(0).get_name();

    let config = builder.create_builder_config();
    // let profile = builder.create_optimization_profile();
    // profile.set_min_dimensions(&input_name, Dims4::new(1, 224, 224, 3));
    // profile.set_opt_dimensions(&input_name, Dims4::new(1, 224, 224, 3));
    // profile.set_max_dimensions(&input_name, Dims4::new(max_batch_size, 224, 224, 3));
    // config.add_optimization_profile(profile);

    config.set_max_workspace_size(workspace_size);

    let binary = builder.serialize(network, config);
    let runtime = Runtime::new(logger);
    let engine = runtime.deserialize_cuda_engine(binary.data().to_vec());

    let input_indice = engine.get_binding_index(&input_name).unwrap();
    println!("{:?}'s indice is {}", input_name, input_indice);

    engine
}

fn main() {
    let logger = Logger::new();
    let file = OnnxFile::new(&PathBuf::from("../assets/resnet200.onnx")).unwrap();
    let engine = create_engine(&logger, file, vec![1], 1 * GB);

    let context = engine.create_execution_context();

    let input_image = image::open("../assets/images/meme.jpg")
        .unwrap()
        .crop(0, 0, 224, 224)
        .into_rgb8();
    eprintln!("Image dimensions: {:?}", input_image.dimensions());

    // Convert image to ndarray
    let array: ndarray_image::NdColor = ndarray_image::NdImage(&input_image).into();
    println!("NdArray len: {}", array.len());
    let mut pre_processed = Array::from_iter(array.iter().map(|&x| 1.0 - (x as f32) / 255.0));

    // Run inference
    let mut output = ndarray::Array1::<f32>::zeros(1000);
    println!("(pre) input: {}", pre_processed);
    println!("(pre) output: {}", output);

    let mut logs = Vec::new();
    let num_of_inference = 1000;
    for _ in 0..num_of_inference {
        let start = std::time::Instant::now();
        context
            .execute_v2(
                ExecuteInput::Float(&mut pre_processed),
                vec![ExecuteInput::Float(&mut output)],
            )
            .unwrap();
        logs.push(start.elapsed());
    }
    logs.sort();
    println!(
        "Latency P50: {:?}",
        logs[(num_of_inference as f32 * 0.50f32) as usize]
    );
    println!(
        "Latency P90: {:?}",
        logs[(num_of_inference as f32 * 0.90f32) as usize]
    );
    println!(
        "Latency P95: {:?}",
        logs[(num_of_inference as f32 * 0.95f32) as usize]
    );
    println!(
        "Latency P99: {:?}",
        logs[(num_of_inference as f32 * 0.99f32) as usize]
    );

    println!("(post) output: {}", output);
}
