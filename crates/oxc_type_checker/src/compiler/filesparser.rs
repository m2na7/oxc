//! Port of typescript-go's `internal/compiler/filesparser.go`.
//!
//! [`FilesParser::parse`] loads all root files and their transitive imports, mirroring tsgo's
//! `filesParser` work-group walk — but driven by rayon (per the request). A single graph thread
//! owns the dedup set and drains results; rayon workers do the parse + import resolution. Dedup is
//! therefore lock-free (single-threaded), following `oxc_linter`'s runtime.

use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    sync::mpsc,
};

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

/// What a worker reports back: a loaded file, or the payload of a panic that occurred while loading
/// it — so the graph thread can re-raise the panic rather than hang.
enum WorkerResult {
    Loaded(LoadedFile),
    Panicked(Box<dyn Any + Send>),
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
            let (tx, rx) = mpsc::channel::<WorkerResult>();
            // `encountered` dedups by normalized path; owned solely by this (graph) thread, so no
            // locking. `pending` counts spawned-but-not-yet-collected tasks; every worker reports
            // back exactly once (even on panic), so it always reaches 0.
            let mut encountered = FxHashSet::<PathBuf>::default();
            let mut pending = 0usize;

            // Seed the roots.
            for root in root_files {
                let path = tspath::to_path(loader.current_directory(), root);
                if encountered.insert(path.clone()) {
                    pending += 1;
                    spawn_load(scope, loader, path, tx.clone());
                }
            }

            // Drain results, enqueuing newly-discovered dependencies. While the channel is empty the
            // graph thread donates itself to the pool via `yield_now` instead of idling.
            while pending > 0 {
                let result = match rx.try_recv() {
                    Ok(result) => result,
                    Err(mpsc::TryRecvError::Empty) => {
                        rayon::yield_now();
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => break,
                };
                pending -= 1;
                let file = match result {
                    WorkerResult::Loaded(file) => file,
                    // A worker panicked (e.g. a parser bug on some file). Re-raise it here rather
                    // than spinning the drain loop forever or silently dropping the file.
                    WorkerResult::Panicked(payload) => panic::resume_unwind(payload),
                };
                for (_specifier, dep_path) in &file.resolved {
                    if encountered.insert(dep_path.clone()) {
                        pending += 1;
                        spawn_load(scope, loader, dep_path.clone(), tx.clone());
                    }
                }
                loaded.push(file);
            }
        });
        loaded
    }
}

/// Spawn a rayon worker that loads `path` and reports back to the graph thread exactly once — the
/// loaded file, or the panic payload if loading panicked. Reporting on panic (instead of just
/// unwinding the worker) keeps `pending` accurate and lets the graph thread re-raise the panic,
/// rather than the drain loop hanging on a task that never reports.
fn spawn_load<'scope>(
    scope: &rayon::Scope<'scope>,
    loader: &'scope FileLoader,
    path: PathBuf,
    tx: mpsc::Sender<WorkerResult>,
) {
    scope.spawn(move |_| {
        let result = match panic::catch_unwind(AssertUnwindSafe(|| ParseTask { path }.load(loader)))
        {
            Ok(file) => WorkerResult::Loaded(file),
            Err(payload) => WorkerResult::Panicked(payload),
        };
        // The receiver is only gone once the graph thread has stopped draining (it is finishing, or
        // itself unwinding a re-raised panic), so a failed send here can be dropped.
        let _ = tx.send(result);
    });
}
