//! No-audio adapter for the upstream key converter.
//!
//! This CLI only supports the 6K-10K layouts, which do not use audio accents.
//! `apply_with(..., None)` follows Torosu's native map-only conversion path.

#[derive(Clone, Copy)]
pub struct MapAccents<'a>(std::marker::PhantomData<&'a ()>);

impl MapAccents<'_> {
    pub fn band_onsets(&self, _time: f64) -> [f32; 3] {
        [0.0; 3]
    }

    pub fn loudness(&self, _from: f64, _to: f64) -> Option<f32> {
        None
    }
}
