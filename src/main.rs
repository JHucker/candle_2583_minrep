use candle_core::Module as CdlModule;
use std::collections::HashMap;
use std::time::Instant;
use tch::nn::Module as TchModule;

fn bench_tch(
    seq_len: i64,
    normalised_shape: i64,
    batch_sizes: &[usize],
    repeats: usize,
) -> Vec<(usize, u128)> {
    let kind = tch::Kind::Float;
    let tch_dev = tch::Device::Cuda(0);
    let tch_vs = tch::nn::VarStore::new(tch_dev);

    // ws is all 1s, bs is all 0s
    let tch_ln = tch::nn::layer_norm(
        tch_vs.root() / "ln",
        vec![normalised_shape],
        tch::nn::LayerNormConfig::default(),
    );

    // input tensors is mean of 0, std of 1
    let input = tch::Tensor::randn([1, seq_len, normalised_shape], (kind, tch_dev));
    for _ in 0..repeats {
        let _ys = tch_ln.forward(&input);
    }

    let mut results = Vec::new();
    for batch_size in batch_sizes {
        let input = tch::Tensor::rand(
            [
                i64::try_from(*batch_size).unwrap(),
                seq_len,
                normalised_shape,
            ],
            (kind, tch_dev),
        );
        tch::Cuda::synchronize(0);

        let mut timings = Vec::new();
        for _ in 0..repeats {
            let tic = Instant::now();
            let _ys = tch_ln.forward(&input);
            tch::Cuda::synchronize(0);
            let elapsed = tic.elapsed().as_micros();
            timings.push(elapsed);
        }
        let average = timings.iter().sum::<u128>() / timings.len() as u128;
        results.push((*batch_size, average));
    }

    results
}

fn bench_candle(
    seq_len: usize,
    normalised_shape: usize,
    batch_sizes: &[usize],
    repeats: usize,
) -> Vec<(usize, u128)> {
    let dtype = candle_core::DType::F32;
    let cdl_dev = candle_core::Device::cuda_if_available(0).unwrap();

    let ln_map: HashMap<String, candle_core::Tensor> = [
        (
            "weight".to_string(),
            candle_core::Tensor::ones(&[normalised_shape], dtype, &cdl_dev).unwrap(),
        ),
        (
            "bias".to_string(),
            candle_core::Tensor::zeros(&[normalised_shape], dtype, &cdl_dev).unwrap(),
        ),
    ]
    .into_iter()
    .collect();

    let cdl_vb = candle_nn::VarBuilder::new_with_args(Box::new(ln_map), dtype, &cdl_dev);

    let cdl_ln = candle_nn::layer_norm(
        normalised_shape,
        candle_nn::LayerNormConfig::default(),
        cdl_vb,
    )
    .unwrap();
    let input = candle_core::Tensor::randn(0.0, 1.0, &[1, seq_len, normalised_shape], &cdl_dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    for _ in 0..repeats {
        let _ys = cdl_ln.forward(&input);
    }

    let mut results = Vec::new();
    for batch_size in batch_sizes {
        let input = candle_core::Tensor::randn(
            0.0,
            1.0,
            &[*batch_size, seq_len, normalised_shape],
            &cdl_dev,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        cdl_dev.synchronize().unwrap();

        let mut timings = Vec::new();
        for _ in 0..repeats {
            let tic = Instant::now();
            let _ys = cdl_ln.forward(&input).unwrap();
            cdl_dev.synchronize().unwrap();
            let elapsed = tic.elapsed().as_micros();
            timings.push(elapsed);
        }
        let average = timings.iter().sum::<u128>() / timings.len() as u128;
        results.push((*batch_size, average));
    }

    results
}

fn main() {
    let seq_len: usize = 76;
    let normalised_shape: usize = 60;
    let repeats: usize = 50;
    let batch_sizes: Vec<_> = (0..12).map(|x| 2_usize.pow(x)).collect();

    let tch_results = bench_tch(
        i64::try_from(seq_len).unwrap(),
        i64::try_from(normalised_shape).unwrap(),
        &batch_sizes,
        repeats,
    );
    let cdl_results = bench_candle(seq_len, normalised_shape, &batch_sizes, repeats);

    println!("batch_size,tch_μs_average,cdl_μs_average");
    for (tch_res, cdl_res) in tch_results.iter().zip(cdl_results.iter()) {
        let (batch_size, tch_us) = tch_res;
        let (_, cdl_us) = cdl_res;
        println!("{batch_size},{tch_us},{cdl_us}");
    }
}
