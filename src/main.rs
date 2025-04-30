mod count_vectorizer;

use count_vectorizer::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;

fn main() {
    let corpus: Vec<String> = vec![
        // Spams
        "Congratulations! You've won a $1,000 gift card. Claim it right now by clicking the link!".to_string(),
        "Bank: Your account is temporarily locked. Please log in at to secure your account".to_string(),
        "Amount received 2.221 Bitcoins BTC ($18,421 USD) Plese confirm transaction".to_string(),

        // Not spams
        "hello world I don’t respect anybody who can’t tell the difference between Pepsi and Coke. What traditional Korean games are there?".to_string(),
        "Edith could decide if she should paint her teeth or brush her nails. Watching the geriatric men’s softball team brought back memories of 3 yr olds playing t-ball.".to_string(),
        "Standing on one's head at job interviews forms a lasting impression. He learned the hardest lesson of his life and had the scars, both physical and mental, to prove it.".to_string(),
    ];

    let mut vectorizer = CountVector::fit(corpus.clone());

    let x: Vec<Vec<f64>> = corpus
        .iter()
        .map(|i| vectorizer.transform(i.to_string()))
        .collect(); // Transforming each String in corpus, also the input data

    let y = vec![1., 1., 1., 0., 0., 0.]; // Output data, 1. = Spam 0. = Not spam

    let x = DenseMatrix::from_2d_vec(&x); // We need to convert the input data to a DenseMatrix so that we can train it

    // We're using Logistic Regression for the spam detection
    let predictor = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

    let input1 = &DenseMatrix::from_2d_vec(&vec![
        vectorizer.transform("hello world how are you doing".to_string())
    ]);

    let input2 = &DenseMatrix::from_2d_vec(&vec![
        vectorizer.transform("congrats you've won a gift card worth $2000 for FREE!".to_string())
    ]);

    println!("hello world how are you doing => {:?}", predictor.predict(input1).unwrap());
    println!("congrats you've won a gift card worth $2000 for FREE! => {:?}", predictor.predict(input2).unwrap());
}
