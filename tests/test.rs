#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    fn bin_dir() -> PathBuf {
        env::current_exe()
            .ok()
            .map(|mut path| {
                path.pop();
                path.pop();
                path
            })
            .unwrap()
    }

    fn tara_exe() -> String {
        bin_dir()
            .join(format!("tara{}", env::consts::EXE_SUFFIX))
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn parser_tests() {
        lit::run::tests(|config| {
            config.add_search_path("tests/parser/");
            config.add_extension("t");

            config.constants.insert("tara".to_owned(), tara_exe());
        })
        .expect("Parser tests failed");
    }

    #[test]
    fn utir_tests() {
        lit::run::tests(|config| {
            config.add_search_path("tests/utir/");
            config.add_extension("t");

            config.constants.insert("tara".to_owned(), tara_exe());
        })
        .expect("UTIR tests failed");
    }

    #[ignore]
    #[test]
    fn tir_tests() {
        lit::run::tests(|config| {
            config.add_search_path("tests/tir/");
            config.add_extension("t");

            config.constants.insert("tara".to_owned(), tara_exe());
        })
        .expect("TIR tests failed");
    }
}
