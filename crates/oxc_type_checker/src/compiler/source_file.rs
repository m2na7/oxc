//! A parsed source file — its arena, AST, and module record — referenced by [`FileId`].
//!
//! Corresponds to typescript-go's `ast.SourceFile` (`internal/ast/ast.go`) + `SourceFileParseOptions`
//! (`internal/ast/parseoptions.go`). Unlike the previous step (which held only paths), a loaded file
//! now keeps its parsed AST so the checker can run over it without re-parsing. No `Semantic` is built
//! yet — only the parse output (`Program` + `ModuleRecord`), which is what import discovery needs.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use oxc_allocator::Allocator;
use oxc_ast::ast::Program as AstProgram;
use oxc_diagnostics::Diagnostics;
use oxc_parser::Parser;
use oxc_span::SourceType;
use oxc_str::CompactStr;
use oxc_syntax::module_record::ModuleRecord;
use rustc_hash::FxHashMap;
use self_cell::self_cell;

use super::program::FileId;

/// Inputs to parse a source file, mirroring tsgo's `ast.SourceFileParseOptions`.
#[derive(Debug, Clone)]
pub struct SourceFileParseOptions {
    /// The file's resolved name (absolute, normalized).
    pub file_name: PathBuf,
    /// The normalized path used as the file's identity key (tsgo `tspath.Path`).
    pub path: PathBuf,
}

/// Backing storage the AST + module record borrow from: the arena and the owned source text.
struct SourceFileOwner {
    allocator: Allocator,
    source_text: String,
}

// A `SourceFile` is self-referential: the parsed `Program`/`ModuleRecord` borrow from the arena and
// source text. `self_cell` stores the owner (arena + text) alongside the parse output that borrows
// from it. Mirrors `oxc_linter`'s `ModuleContent`.
self_cell! {
    struct SourceFileCell {
        owner: SourceFileOwner,
        #[covariant]
        dependent: SourceFileData,
    }
}

struct SourceFileData<'a> {
    program: AstProgram<'a>,
    module_record: ModuleRecord<'a>,
}

// SAFETY: `SourceFileCell` owns the arena (inside `SourceFileOwner`) together with the `Program` and
// `ModuleRecord` that borrow from it, with no outside borrows. Moving the cell moves the arena with
// its dependents, so the arena references stay valid across threads. This lets a parsed file be sent
// between rayon workers and the graph thread (mirrors `oxc_linter`'s `ModuleContent`).
unsafe impl Send for SourceFileCell {}

/// A single parsed source file.
///
/// Corresponds to tsgo's `*ast.SourceFile`. It keeps its arena-backed AST (`program`) and import
/// data (`module_record`), plus the resolved module-graph edges filled in once every file has a
/// [`FileId`].
pub struct SourceFile {
    parse_options: SourceFileParseOptions,
    source_type: SourceType,
    cell: SourceFileCell,
    /// Parse diagnostics. Owned (they do not borrow the arena). Not yet rendered.
    diagnostics: Diagnostics,
    /// Module-graph edges: each resolved import specifier -> the dependency's [`FileId`]. Populated
    /// after all files are loaded (tsgo `resolutionsInFile`, roughly).
    resolved_modules: FxHashMap<CompactStr, FileId>,
}

impl SourceFile {
    /// Parse `source_text`, mirroring tsgo's `parser.ParseSourceFile`. `source_type` selects the
    /// JS/TS dialect (derived from the file extension).
    pub(crate) fn parse(
        parse_options: SourceFileParseOptions,
        source_text: String,
        source_type: SourceType,
    ) -> Self {
        let mut diagnostics = Diagnostics::new();
        let owner = SourceFileOwner { allocator: Allocator::default(), source_text };
        let cell = SourceFileCell::new(owner, |owner| {
            let ret = Parser::new(&owner.allocator, &owner.source_text, source_type).parse();
            diagnostics.extend(ret.diagnostics.into_vec());
            SourceFileData { program: ret.program, module_record: ret.module_record }
        });
        Self {
            parse_options,
            source_type,
            cell,
            diagnostics,
            resolved_modules: FxHashMap::default(),
        }
    }

    /// The file's resolved name (absolute, normalized).
    pub fn file_name(&self) -> &Path {
        &self.parse_options.file_name
    }

    /// The file's normalized identity key (tsgo `tspath.Path`).
    pub fn path(&self) -> &Path {
        &self.parse_options.path
    }

    /// The JS/TS dialect the file was parsed as.
    pub fn source_type(&self) -> SourceType {
        self.source_type
    }

    /// Parse diagnostics collected for this file.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// The parsed AST.
    pub fn program(&self) -> &AstProgram<'_> {
        &self.cell.borrow_dependent().program
    }

    /// The file's module record (imports/exports).
    pub fn module_record(&self) -> &ModuleRecord<'_> {
        &self.cell.borrow_dependent().module_record
    }

    /// Resolved module-graph edges: import specifier -> dependency [`FileId`].
    pub fn resolved_modules(&self) -> &FxHashMap<CompactStr, FileId> {
        &self.resolved_modules
    }

    pub(super) fn set_resolved_modules(&mut self, resolved: FxHashMap<CompactStr, FileId>) {
        self.resolved_modules = resolved;
    }
}

impl fmt::Debug for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceFile")
            .field("file_name", &self.parse_options.file_name)
            .field("source_type", &self.source_type)
            .field("diagnostics", &self.diagnostics.len())
            .field("resolved_modules", &self.resolved_modules.len())
            .finish_non_exhaustive()
    }
}
