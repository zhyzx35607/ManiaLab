pub mod commands {
    pub mod trainer {
        pub mod accents;

        pub mod analysis {
            pub mod layout;
            pub mod patterns;
            pub mod zones;
        }

        pub mod mania;
        pub mod model;
        pub mod timing;

        pub mod tasks {
            pub mod fln_converter;
            pub mod key_converter;
            pub mod short_ln;
        }
    }
}

pub mod error;
pub mod osu_fixups;
