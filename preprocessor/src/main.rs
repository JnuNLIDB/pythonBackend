use console::{style, Emoji};
use indicatif::{ProgressIterator, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, Write};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct Source {
    #[serde(rename = "Name")]
    name: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct JsonData {
    #[serde(rename = "Headline")]
    headline: String,
    #[serde(rename = "Body")]
    body: Option<String>,
    #[serde(rename = "Source")]
    source: Vec<Source>,
}

impl Hash for JsonData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.body.hash(state);
    }
}

impl PartialEq for JsonData {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

impl Eq for JsonData {}

fn read_data_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<JsonData>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

fn chunk_body(body: &str, size: usize) -> Vec<&str> {
    // If the body is larger than size, try splitting it into chunks
    if body.len() > size {
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut end = size;
        while end < body.len() {
            // Find the last space in the chunk
            while !body.is_char_boundary(end) {
                end -= 1;
            }
            let last_space = match body[start..end].rfind(' ') {
                None => end,
                Some(k) => k + start,
            };
            chunks.push(&body[start..last_space]);
            start = last_space;
            end = start + size;
        }
        chunks.push(&body[start..]);
        chunks
    } else {
        vec![body]
    }
}

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍 ", "");
static TRUCK: Emoji<'_, '_> = Emoji("🚚 ", "");
static PEN: Emoji<'_, '_> = Emoji("🖊 ", "");
static SPARKLE: Emoji<'_, '_> = Emoji("✨ ", "");

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| {
        println!("Usage: cargo run --release -- <name>");
        std::process::exit(1);
    });
    let original_path = format!("../data/{}.json", arg);
    let path = Path::new(&original_path);
    if !path.exists() {
        println!("File {} does not exist", original_path);
        std::process::exit(1);
    }

    let save_path = format!("../data/{}_preprocessed.txt", arg);
    let save_path = Path::new(&save_path);

    let bar_style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");

    println!(
        "{} {}Loading Data in file...",
        style("[1/4]").bold().dim(),
        LOOKING_GLASS
    );
    let data = read_data_from_file(path).unwrap();

    // Filter those that are not unique
    println!(
        "{} {}Filtering unique data...",
        style("[2/4]").bold().dim(),
        TRUCK
    );
    let hashset = data
        .into_iter()
        .progress()
        .with_style(bar_style.clone())
        .filter(|x| x.source[0].name.is_some())
        .collect::<std::collections::HashSet<_>>();

    // Write file
    println!("{} {}Writing to file...", style("[3/4]").bold().dim(), PEN);
    let length = hashset.len() / 10;
    let bar = indicatif::ProgressBar::new(length as u64);
    bar.set_style(bar_style);

    let file = File::create(save_path).unwrap();
    let mut counter = 0;
    for json_data in hashset.into_iter().take(length) {
        let Some(body) = json_data.body else { continue; };
        let body = body
            .replace(' ', " ")
            .split('\n')
            .filter(|x| !x.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let chunked_body = chunk_body(&body, 2048);
        for b in chunked_body {
            writeln!(
                &file,
                "{} By {}: \n{}\n",
                json_data.headline,
                json_data.source[0].name.as_ref().unwrap(),
                b
            )
            .unwrap();
            counter += 1;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    println!(
        "{} {}Finished preprocessing {} data",
        style("[4/4]").bold().dim(),
        SPARKLE,
        counter
    );
}
