use ndarray::Array2;
use rusty_machine::learning::gp::GaussianProcess;
use rusty_machine::learning::gp::ConstMean;
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::error::Error;

#[derive(Serialize, Deserialize)]
struct Params {
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    subsample: f64,
    colsample_bytree: f64,
}

fn load_data(file_path: &str) -> Result<(Matrix<f64>, Matrix<f64>), Box<dyn Error>> {
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

    let inputs = data_matrix.slice(0..rows, 0..cols-1);
    let targets = data_matrix.slice(0..rows, cols-1..cols);

    Ok((inputs, targets))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (inputs, targets) = load_data("data.json")?;

    let kernel = kernel::SquaredExp::default();
    let mut gp = GaussianProcess::new(kernel, ConstMean::default());

    gp.train(&inputs, &targets)?;

    let preds = gp.predict(&inputs)?;
    println!("Predictions: {:?}", preds);

    Ok(())
}
