//! Paths shared across volkit: its per-user cache root and canonical paths without Windows' verbatim prefix

use std::path::{Path, PathBuf};

/// volkit's cache directory: `$VOLKIT_CACHE_DIR`, else `$XDG_CACHE_HOME/volkit`, else `~/.cache/volkit`
/// (`%LOCALAPPDATA%\volkit` on Windows); None when no directory can be named
pub fn cache_root() -> Option<PathBuf> {
    cache_root_in(&|name| std::env::var_os(name))
}

/// [`cache_root`] as `env` names it, so a caller can resolve it from an environment of its own
pub fn cache_root_in(env: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    let nonempty = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    nonempty("VOLKIT_CACHE_DIR")
        .or_else(|| nonempty("XDG_CACHE_HOME").map(|p| p.join("volkit")))
        .or_else(|| default_cache_home(&nonempty).map(|p| p.join("volkit")))
}

/// The per-user cache directory when `XDG_CACHE_HOME` is unset: `~/.cache`
#[cfg(not(windows))]
fn default_cache_home(nonempty: &dyn Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    nonempty("HOME").map(|home| home.join(".cache"))
}

/// The per-user cache directory when `XDG_CACHE_HOME` is unset
///
/// Windows sets no `HOME`. cuDNN plans and compiled kernels belong to this machine's GPU, so they go in the local
/// profile, `%LOCALAPPDATA%`, not the one that roams to other machines
#[cfg(windows)]
fn default_cache_home(nonempty: &dyn Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    nonempty("LOCALAPPDATA")
}

/// `std::fs::canonicalize`, without the `\\?\` prefix Windows adds when the path reads the same without it
///
/// Python's `Path.resolve()` gives paths without the prefix, so a path volkit keys or compares on spells them the
/// same way. A path that needs the prefix, such as one with a trailing dot, keeps it
pub fn canonicalize(path: &Path) -> std::io::Result<PathBuf> {
    let full = std::fs::canonicalize(path)?;
    #[cfg(windows)]
    let full = without_verbatim_prefix(full);
    Ok(full)
}

/// `full` (a canonical path) as `C:\...` or `\\server\share\...` when that names the same file
#[cfg(windows)]
fn without_verbatim_prefix(full: PathBuf) -> PathBuf {
    let Some(text) = full.to_str() else {
        return full;
    };
    let plain = if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\")
        && rest.as_bytes().get(1) == Some(&b':')
    {
        rest.to_owned()
    } else {
        return full;
    };
    // the plain form goes through Windows' path normalisation, which the verbatim one skips
    match std::fs::canonicalize(&plain) {
        Ok(again) if again == full => PathBuf::from(plain),
        _ => full,
    }
}
