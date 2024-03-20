use kioku::Arena;

pub struct Package<'arena> {
    pub src_dir: &'arena str,
    pub src_path: &'arena str,
    pub pkg_path: &'arena str,
}

impl<'arena> Package<'arena> {
    pub fn new_in(arena: &'arena Arena, src_dir: &str, src_path: &str, pkg_path: &str) -> Self {
        let src_dir = arena.copy_str(src_dir);
        let src_path = arena.copy_str(src_path);
        let pkg_path = arena.copy_str(pkg_path);
        Self {
            src_dir,
            src_path,
            pkg_path,
        }
    }

    pub fn full_path(&self) -> String {
        let mut s = String::new();
        s.push_str(self.src_dir);
        s.push_str(std::path::MAIN_SEPARATOR_STR);
        s.push_str(self.src_path);
        s
    }
}
