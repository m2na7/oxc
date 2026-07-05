//! Port of typescript-go's `internal/compiler/fileloader.go`.
//!
//! [`FileLoader::process_all_program_files`] (tsgo `processAllProgramFiles`) parses the root files,
//! follows their imports to load the dependent files in parallel, and collects everything into
//! [`ProcessedFiles`]. Import resolution ([`FileLoader::resolve_imports`]) mirrors
//! `resolveImportsAndModuleAugmentations`, reusing `oxc_resolver`'s TS-aware `resolve_dts`.

use std::path::{Path, PathBuf};

use oxc_index::IndexVec;
use oxc_resolver::{
    ResolveOptions, Resolver, TsconfigDiscovery, TsconfigOptions, TsconfigReferences,
};
use oxc_span::VALID_EXTENSIONS;
use oxc_str::CompactStr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::tspath;

use super::{
    filesparser::{FilesParser, LoadedFile},
    host::CompilerHost,
    program::FileId,
    source_file::SourceFile,
};

/// A program's loaded files — the subset of tsgo's `processedFiles` this step fills.
///
/// `files` is the parsed files in a deterministic order; `files_by_path` maps normalized paths to
/// ids; `missing_files` are root/import paths that could not be read.
#[derive(Debug, Default)]
pub(super) struct ProcessedFiles {
    /// Parsed files (tsgo `files`).
    pub(super) files: IndexVec<FileId, SourceFile>,
    /// Normalized path -> file id (tsgo `filesByPath`).
    pub(super) files_by_path: FxHashMap<PathBuf, FileId>,
    /// Paths that produced no source file (tsgo `missingFiles`).
    pub(super) missing_files: Vec<PathBuf>,
}

/// Loads a program's files, mirroring tsgo's `fileLoader`. Holds the host (reads + parses) and the
/// module resolver. `Send + Sync`, so rayon workers share one by `&`.
pub(super) struct FileLoader {
    host: CompilerHost,
    resolver: Resolver,
}

impl FileLoader {
    fn new(current_directory: PathBuf, tsconfig: Option<&Path>) -> Self {
        Self { host: CompilerHost::new(current_directory), resolver: build_resolver(tsconfig) }
    }

    pub(super) fn host(&self) -> &CompilerHost {
        &self.host
    }

    pub(super) fn current_directory(&self) -> &Path {
        self.host.current_directory()
    }

    /// tsgo `resolveImportsAndModuleAugmentations`: resolve each of the file's import specifiers to
    /// a normalized dependency path. Specifiers that don't resolve (e.g. uninstalled node_modules)
    /// are skipped.
    pub(super) fn resolve_imports(&self, source_file: &SourceFile) -> Vec<(CompactStr, PathBuf)> {
        let importing_file = source_file.file_name();
        source_file
            .module_record()
            .requested_modules
            .keys()
            .filter_map(|specifier| {
                let specifier = specifier.as_str();
                let resolution = self.resolver.resolve_dts(importing_file, specifier).ok()?;
                let path = tspath::to_path(self.current_directory(), resolution.path());
                Some((CompactStr::from(specifier), path))
            })
            .collect()
    }

    /// tsgo `processAllProgramFiles`: parse the roots, load their transitive imports in parallel,
    /// and collect the result.
    pub(super) fn process_all_program_files(
        current_directory: &Path,
        root_files: &[PathBuf],
        tsconfig: Option<&Path>,
    ) -> ProcessedFiles {
        let loader = Self::new(current_directory.to_path_buf(), tsconfig);
        let loaded = FilesParser::parse(&loader, root_files);
        loader.collect(root_files, loaded)
    }

    /// tsgo `filesParser.getProcessedFiles`/`collectFiles`: assign each loaded file a deterministic
    /// [`FileId`] (roots in include order, then the rest sorted by path) and link the module-graph
    /// edges. Rayon yields files in nondeterministic order, so ordering happens here.
    fn collect(&self, root_files: &[PathBuf], loaded: Vec<LoadedFile>) -> ProcessedFiles {
        let mut by_path: FxHashMap<PathBuf, LoadedFile> =
            loaded.into_iter().map(|file| (file.path.clone(), file)).collect();

        // Deterministic order: roots (include order) first, then the remaining paths sorted.
        let mut ordered: Vec<PathBuf> = Vec::with_capacity(by_path.len());
        let mut seen = FxHashSet::<PathBuf>::default();
        for root in root_files {
            let path = tspath::to_path(self.current_directory(), root);
            if by_path.contains_key(&path) && seen.insert(path.clone()) {
                ordered.push(path);
            }
        }
        let mut rest: Vec<PathBuf> =
            by_path.keys().filter(|path| !seen.contains(*path)).cloned().collect();
        rest.sort_unstable();
        ordered.extend(rest);

        // Assign FileIds; stash each file's resolved edges alongside (aligned with FileId order).
        let mut processed = ProcessedFiles::default();
        let mut resolved_by_id: Vec<Vec<(CompactStr, PathBuf)>> = Vec::new();
        for path in ordered {
            let LoadedFile { source_file, resolved, .. } = by_path.remove(&path).unwrap();
            match source_file {
                Some(source_file) => {
                    let id = processed.files.push(source_file);
                    processed.files_by_path.insert(path, id);
                    resolved_by_id.push(resolved);
                }
                None => processed.missing_files.push(path),
            }
        }

        // Link edges: import specifier -> dependency FileId (dropping deps that failed to load).
        for (index, resolved) in resolved_by_id.into_iter().enumerate() {
            let edges = resolved
                .into_iter()
                .filter_map(|(specifier, dep_path)| {
                    processed.files_by_path.get(&dep_path).map(|&id| (specifier, id))
                })
                .collect();
            processed.files[FileId::from_usize(index)].set_resolved_modules(edges);
        }
        processed
    }
}

/// Build the module resolver, mirroring `oxc_linter`'s `get_resolver` but for TS declaration
/// resolution (`resolve_dts`): TS extensions, `.js`->`.ts` aliasing, and the project's tsconfig
/// (for `paths`/`baseUrl`).
fn build_resolver(tsconfig: Option<&Path>) -> Resolver {
    let tsconfig = tsconfig.map(|path| {
        TsconfigDiscovery::Manual(TsconfigOptions {
            config_file: path.to_path_buf(),
            references: TsconfigReferences::Auto,
        })
    });
    Resolver::new(ResolveOptions {
        extensions: VALID_EXTENSIONS.iter().map(|ext| format!(".{ext}")).collect(),
        main_fields: vec!["module".to_string(), "main".to_string()],
        condition_names: vec!["module".to_string(), "import".to_string()],
        extension_alias: vec![
            (".js".to_string(), vec![".js".to_string(), ".ts".to_string()]),
            (".mjs".to_string(), vec![".mjs".to_string(), ".mts".to_string()]),
            (".cjs".to_string(), vec![".cjs".to_string(), ".cts".to_string()]),
        ],
        tsconfig,
        ..ResolveOptions::default()
    })
}
