//! Port of typescript-go's `internal/compiler/filesparser.go`.
//!
//! [`FilesParser::parse`] loads all root files and their transitive imports, mirroring tsgo's
//! `filesParser` work-group walk — but driven by rayon (per the request). A single graph thread
//! owns the dedup set and drains results; rayon workers do the parse + import resolution. Dedup is
//! therefore lock-free (single-threaded), following `oxc_linter`'s runtime.

use std::{path::PathBuf, sync::mpsc};

use oxc_str::CompactStr;
use rustc_hash::FxHashSet;

use crate::tspath;

use super::{
    fileloader::FileLoader,
    source_file::{SourceFile, SourceFileParseOptions},
};

/// A file loaded by a worker: its parsed [`SourceFile`] (`None` if unreadable) and the resolved
/// paths of its imports (tsgo `parseTask` after `load` + `resolveImportsAndModuleAugmentations`).
pub(super) struct LoadedFile {
    pub(super) path: PathBuf,
    pub(super) source_file: Option<SourceFile>,
    /// Import specifier -> resolved dependency path (normalized).
    pub(super) resolved: Vec<(CompactStr, PathBuf)>,
}

/// A unit of work: load one file. Mirrors tsgo's `parseTask`.
struct ParseTask {
    path: PathBuf,
}

impl ParseTask {
    /// tsgo `parseTask.load`: read + parse the file, then resolve its imports to dependency paths.
    fn load(self, loader: &FileLoader) -> LoadedFile {
        let opts = SourceFileParseOptions { file_name: self.path.clone(), path: self.path.clone() };
        match loader.host().get_source_file(opts) {
            Some(source_file) => {
                let resolved = loader.resolve_imports(&source_file);
                LoadedFile { path: self.path, source_file: Some(source_file), resolved }
            }
            None => LoadedFile { path: self.path, source_file: None, resolved: Vec::new() },
        }
    }
}

/// Drives parallel loading, mirroring tsgo's `filesParser`.
pub(super) struct FilesParser;

impl FilesParser {
    /// Load `root_files` and every file reachable through their imports, in parallel. Returns each
    /// loaded file in nondeterministic arrival order — the caller assigns [`FileId`](super::FileId)s
    /// deterministically.
    pub(super) fn parse(loader: &FileLoader, root_files: &[PathBuf]) -> Vec<LoadedFile> {
        let mut loaded = Vec::new();
        rayon::scope(|scope| {
            let (tx, rx) = mpsc::channel::<LoadedFile>();
            // `encountered` dedups by normalized path; owned solely by this (graph) thread, so no
            // locking. `pending` counts spawned-but-not-yet-collected tasks.
            let mut encountered = FxHashSet::<PathBuf>::default();
            let mut pending = 0usize;

            // Seed the roots.
            for root in root_files {
                let path = tspath::to_path(loader.current_directory(), root);
                if encountered.insert(path.clone()) {
                    pending += 1;
                    let tx = tx.clone();
                    scope.spawn(move |_| tx.send(ParseTask { path }.load(loader)).unwrap());
                }
            }

            // Drain results, enqueuing newly-discovered dependencies. While the channel is empty the
            // graph thread donates itself to the pool via `yield_now` instead of idling.
            while pending > 0 {
                let file = match rx.try_recv() {
                    Ok(file) => file,
                    Err(mpsc::TryRecvError::Empty) => {
                        rayon::yield_now();
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => break,
                };
                pending -= 1;
                for (_specifier, dep_path) in &file.resolved {
                    if encountered.insert(dep_path.clone()) {
                        pending += 1;
                        let tx = tx.clone();
                        let path = dep_path.clone();
                        scope.spawn(move |_| tx.send(ParseTask { path }.load(loader)).unwrap());
                    }
                }
                loaded.push(file);
            }
        });
        loaded
    }
}
