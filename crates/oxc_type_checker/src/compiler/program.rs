//! Port of typescript-go's `internal/compiler/program.go`.

use std::path::{Path, PathBuf};

use oxc_index::{IndexSlice, define_nonmax_u32_index_type};

use super::{
    fileloader::{FileLoader, ProcessedFiles},
    source_file::SourceFile,
};

define_nonmax_u32_index_type! {
    /// Index of a [`SourceFile`] within a [`Program`].
    ///
    /// typescript-go has no integer file id — it keys files by their normalized `tspath.Path`.
    /// This typed index is an oxc-side addition so files can be referenced by a cheap `u32` (and,
    /// later, declarations by `(FileId, SymbolId)`).
    pub struct FileId;
}

/// A program: the source files loaded from a set of root files (roots + their transitive imports).
/// Mirrors tsgo's `Program` (which embeds `processedFiles`).
///
/// This is the in-memory model the type checker will run over. Files are parsed (AST + module
/// record) but not yet bound; type checking is a later step.
#[derive(Debug)]
pub struct Program {
    processed: ProcessedFiles,
}

impl Program {
    /// Port of tsgo's `NewProgram`: parse `root_files`, follow their imports to load every
    /// dependent file (in parallel), and collect them. Relative paths resolve against
    /// `current_directory`; `tsconfig` (when present) drives module resolution (`paths`/`baseUrl`).
    pub fn new(current_directory: &Path, root_files: &[PathBuf], tsconfig: Option<&Path>) -> Self {
        Self {
            processed: FileLoader::process_all_program_files(
                current_directory,
                root_files,
                tsconfig,
            ),
        }
    }

    /// All source files, in deterministic order (tsgo `Program.SourceFiles`).
    pub fn files(&self) -> &IndexSlice<FileId, [SourceFile]> {
        &self.processed.files
    }

    /// The source file with the given [`FileId`].
    pub fn file(&self, id: FileId) -> &SourceFile {
        &self.processed.files[id]
    }

    /// The [`FileId`] for a normalized path, if the program contains it (tsgo
    /// `Program.GetSourceFileByPath`).
    pub fn file_id(&self, path: &Path) -> Option<FileId> {
        self.processed.files_by_path.get(path).copied()
    }

    /// Paths that could not be read (tsgo `missingFiles`).
    pub fn missing_files(&self) -> &[PathBuf] {
        &self.processed.missing_files
    }

    /// The number of source files.
    pub fn len(&self) -> usize {
        self.processed.files.len()
    }

    /// Whether the program has no source files.
    pub fn is_empty(&self) -> bool {
        self.processed.files.is_empty()
    }
}
