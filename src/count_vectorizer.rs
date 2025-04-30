use regex::Regex;
use std::collections::HashMap;

pub struct CountVector {
    dictionary: Vec<String>,
}

fn clean_string(text: String) -> String { // Removes punctuations, numbers, etc. from String
    Regex::new(r"[^a-zA-Z ]")
        .unwrap()
        .replace_all(&text.to_lowercase(), "")
        .to_string()
}

impl CountVector {
    pub fn fit(x: Vec<String>) -> CountVector { // Adds words to dictionary
        let mut dict = Vec::<String>::new(); // The dictionary to add words to
        
        for text in x { // Since the corpus will be a vector of Strings, we need to iterate over it
            for word in clean_string(text).split(' ') {
                if !dict.iter().any(|i| i == word) { // Checking if a word exists in dict
                    dict.push(word.to_string()); // If not, then push the word to dict
                }
            }
        }

        CountVector { dictionary: dict }
    }

    pub fn transform(&mut self, text: String) -> Vec<f64> { // Converts a string to a vector of f64s that contains the count of words
        let mut count: HashMap<String, f64> = HashMap::new();
        let mut out = Vec::<f64>::new();

        for i in &self.dictionary {
            count.insert(i.clone(), 0.);
        }

        for i in clean_string(text).split(' ') {
            if count.contains_key(i) {
                *count.get_mut(i).unwrap() += 1.0;
            }
        }

        for i in &self.dictionary {
            out.push(count[i]);
        }

        out
    }
}