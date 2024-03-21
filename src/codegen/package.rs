pub struct Package {
    pub src_dir: String,
    pub src_path: String,
    pub pkg_path: String,
}

impl Package {
    pub fn new_in(src_dir: String, src_path: String, pkg_path: String) -> Self {
        Self {
            src_dir,
            src_path,
            pkg_path,
        }
    }

    pub fn full_path(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.src_dir);
        s.push_str(std::path::MAIN_SEPARATOR_STR);
        s.push_str(&self.src_path);
        s
    }
}
