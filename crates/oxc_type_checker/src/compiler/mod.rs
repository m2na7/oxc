//! Port of typescript-go's `internal/compiler` package.
//!
//! Builds a [`Program`] from a project's root files: parse each file, follow its module-record
//! imports to load the dependent files (in parallel via rayon), and collect them into an index-vec
//! keyed by [`FileId`]. File reading, path normalization, and module resolution reuse
//! `oxc_resolver`. Binding and type checking are later steps.

mod fileloader;
mod filesparser;
mod host;
mod program;
mod source_file;

pub use program::{FileId, Program};
pub use source_file::SourceFile;
