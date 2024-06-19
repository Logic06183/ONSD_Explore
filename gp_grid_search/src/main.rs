use rusty_machine::learning::gp::GaussianProcess;
use rusty_machine::learning::gp::ConstMean;
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::{Matrix, Vector};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::error::Error;
use rusty_machine::prelude::BaseMatrix;

#[derive(Serialize, Deserialize)]
struct Params {
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    subsample: f64,
    colsample_bytree: f64,
}

fn load_data(file_path: &str) -> Result<(Matrix<f64>, Vector<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let data: Vec<Vec<f64>> = serde_json::from_reader(reader)?;

    let rows = data.len();
    let cols = data[0].len();
    let mut data_matrix = Matrix::zeros(rows, cols);

    for (i, row) in data.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            data_matrix[[i, j]] = val;
        }
    }

    let input_cols: Vec<usize> = (0..cols-1).collect();
    let target_cols: Vec<usize> = vec![cols-1];

    let inputs = data_matrix.select_cols(&input_cols);
    let targets = data_matrix.select_cols(&target_cols).into_vec();

    Ok((inputs, Vector::new(targets)))
}

fn evaluate_model(inputs: &Matrix<f64>, targets: &Vector<f64>, kernel: kernel::SquaredExp) -> f64 {
    let mut gp = GaussianProcess::new(kernel, ConstMean::default(), 1.0);

    // Train the model
    gp.train(inputs, targets).unwrap();

    // Predict
    let preds = gp.predict(inputs).unwrap();

    // Calculate the mean squared error
    let mut mse = 0.0;
    for i in 0..targets.size() {
        mse += (targets[i] - preds[i]).powi(2);
    }
    mse / (targets.size() as f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (inputs, targets) = load_data("data.json")?;

    // Define the grid of hyperparameters
    let lscales = vec![0.1, 1.0, 10.0];
    let sigmas = vec![0.1, 1.0, 10.0];

    let mut best_mse = f64::INFINITY;
    let mut best_params = (0.0, 0.0);

    for &lscale in &lscales {
        for &sigma in &sigmas {
            let kernel = kernel::SquaredExp::new(lscale, sigma);
            let mse = evaluate_model(&inputs, &targets, kernel);
            println!("lscale: {}, sigma: {}, mse: {}", lscale, sigma, mse);

            if mse < best_mse {
                best_mse = mse;
                best_params = (lscale, sigma);
            }
        }
    }

    println!("Best params - lscale: {}, sigma: {}", best_params.0, best_params.1);
    println!("Best MSE: {}", best_mse);

    Ok(())
}
