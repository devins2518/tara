pub struct Tld {
    kind: TldKind,
    status: TldResolveStatus,
}

pub enum TldKind {
    Fn,
    Container,
}

pub enum TldResolveStatus {
    Unresolved,
    InProgress,
    Failed,
    Resolved,
}
