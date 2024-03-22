use std::path::PathBuf;

pub struct Package {
    pub src_dir: PathBuf,
    pub src_path: String,
    pub pkg_path: String,
}

impl Package {
    pub fn new_in(src_dir: PathBuf, src_path: String, pkg_path: String) -> Self {
        Self {
            src_dir,
            src_path,
            pkg_path,
        }
    }

    pub fn full_path(&self) -> PathBuf {
        self.src_dir.join(&self.src_path)
    }
}
