use csv::Error;
use rand::distributions::{Distribution, Uniform};
use clap::{Arg, App};

fn load_csv(filename : String) -> Result<Vec<Vec<f64>>,Error>{
    
    let mut dataset:Vec<Vec<f64>> = vec![];
    let mut reader = csv::Reader::from_path(filename)?;

    for record in reader.records(){
        let mut temp : Vec<f64> = vec![];
        let line = record?;
        for val in &line{
            temp.push(val.parse::<f64>().unwrap())
        }
        dataset.push(temp)
    }
    Ok(dataset)
}

fn train_test_split(dataset: &Vec<Vec<f64>>, labels: &Vec<Vec<f64>>, ratio : f64) -> [Vec<Vec<f64>>; 4] {
    let mut dataset_test:Vec<Vec<f64>> = vec![];
    let mut labels_test:Vec<Vec<f64>> = vec![];
    let mut dataset_train = dataset.clone();
    let mut labels_train = labels.clone();

    let mut rng = rand::thread_rng();

    loop {
        let index = Uniform::from(0..dataset_train.len()).sample(&mut rng);
        dataset_test.push(dataset_train.remove(index));
        labels_test.push(labels_train.remove(index));
        if dataset_test.len() as f64 >= dataset.len() as f64 * ratio{
            break;
        }
    }

    [dataset_train,labels_train,dataset_test,labels_test]
}

// Calculate accuracy percentage
fn  accuracy_metric(actual:&Vec<Vec<f64>>, predicted:&Vec<f64>)->f64 {
    let mut correct = 0;
    for i in 0..actual.len() {
        if actual[i][0] == predicted[i] {
            correct += 1;
        }
    }
    correct as f64/ (actual.len() as f64) * 100.0
}
// Make a prediction with coefficients
fn predict(row:&Vec<f64>, coefficients:&Vec<f64>) -> f64 {
    let mut yhat = 0.0;
    for i in 0..row.len() {
        yhat += coefficients[i] * row[i];
    }
    1.0 / (1.0 + f64::exp(-yhat))
}
// Estimate logistic regression coefficients using stochastic gradient descent
fn coefficients_sgd(x_train:&Vec<Vec<f64>>, y_train:&Vec<Vec<f64>>, l_rate:f64, n_epoch:i32) -> Vec<f64> {
    let mut coef:Vec<f64> = vec![0.0; x_train[0].len()];
    
    for _ in 0..n_epoch {
        for (i,  row) in x_train.iter().enumerate(){
            let yhat = predict(row, &coef);
            let error = y_train[i][0] - yhat;
            for i in 0..row.len(){
                coef[i] = coef[i] + l_rate * error * yhat * (1.0 - yhat) * row[i];
            }
        }
    }
    coef
}

// Linear Regression Algorithm With Stochastic Gradient Descent
fn logistic_regression(x_train:&Vec<Vec<f64>>, y_train:&Vec<Vec<f64>>, x_test:&Vec<Vec<f64>>, y_test:&Vec<Vec<f64>>, l_rate:f64, n_epoch:i32) -> f64 {
    let mut predictions: Vec<f64> = vec![];
    let coef = coefficients_sgd(x_train, y_train, l_rate, n_epoch);

    for row in x_test {
        let mut yhat = predict(&row, &coef);
        yhat = f64::round(yhat);
        predictions.push(yhat);
    }
    accuracy_metric(&y_test, &predictions)
}

fn main() -> Result<(),Error> {
    let matches = App::new("")
        .arg(Arg::with_name("n_size").long("n_size").takes_value(true))
        .arg(Arg::with_name("l_rate").long("l_rate").takes_value(true))
        .arg(Arg::with_name("n_epoch").long("n_epoch").takes_value(true))
        .get_matches();

    let n_size = matches.value_of("n_size").unwrap_or("1000").parse::<i32>().unwrap();
    let l_rate = matches.value_of("l_rate").unwrap_or("0.5").parse::<f64>().unwrap();
    let n_epoch = matches.value_of("n_epoch").unwrap_or("5").parse::<i32>().unwrap();
    let ratio = 0.3;

    let data_file = format!("data/moon_data_{}.csv",n_size);
    let label_file = format!("data/moon_labels_{}.csv",n_size);

    let dataset = load_csv(data_file)?;
    let labels = load_csv(label_file)?;

    let data = train_test_split(&dataset,&labels,ratio);
    let score = logistic_regression(&data[0],&data[1],&data[2],&data[3],l_rate,n_epoch);

    println!("{}",score);
    Ok(())
}
