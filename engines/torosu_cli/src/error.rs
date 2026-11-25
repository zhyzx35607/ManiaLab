use std::fmt;

#[derive(Debug)]
pub enum Error {
    ManiaOnly { task: &'static str },
    UnsupportedKeys { keys: usize },
    OnePlayer { layout: String },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManiaOnly { task } => write!(formatter, "{task} only supports osu!mania"),
            Self::UnsupportedKeys { keys } => write!(formatter, "unsupported key count: {keys}"),
            Self::OnePlayer { layout } => write!(formatter, "unsupported co-op layout: {layout}"),
        }
    }
}

impl std::error::Error for Error {}
