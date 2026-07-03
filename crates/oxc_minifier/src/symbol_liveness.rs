//! Whole-program symbol liveness for unused-declaration removal (#13105).
//!
//! `Scoping::symbol_is_unused` is reference-count based, so a declaration
//! that only references itself (`function f() { f() }`) or participates in a
//! reference cycle (`function c() { d() } function d() { c() }`) never reaches
//! zero references and survives even though no live code can reach it.
//!
//! This module computes reachability instead: roots are references that occur
//! in live-executing context (including `export { f }` specifiers, which carry
//! real references); references inside a *candidate* declaration's deferred
//! region (function bodies, side-effect-free initializers, class bodies) only
//! mark their targets live once the candidate itself is marked live. Dead
//! cycles are simply never reached — no cycle detection is needed.
//!
//! ## The gate-mirror invariant (correctness, not style)
//!
//! Candidacy here must be a SUBSET of what the removal sites in
//! `remove_unused_declaration.rs` will actually remove — cleanly, with no
//! residue. Every member of a dead cycle is removed in the same pass; if one
//! member's removal were blocked (or left residue carrying references), the
//! survivors would reference bindings that no longer exist. Concretely:
//!
//! - global gates: `unused != Keep`; skip entirely when the root scope has
//!   `ScopeFlags::DirectEval` (mirrors `can_remove_unused_declarators`; the
//!   flag propagates to the root from any direct eval in the program, so this
//!   also subsumes the per-site current-scope checks).
//! - script-mode: statements visited at the root scope are non-candidates
//!   (mirrors `keep_top_level_var_in_script_mode`, which compares the
//!   *visitation* scope, not the symbol's scope).
//! - declarator inits must be dropped whole by `remove_unused_expression`:
//!   kinds with specialized handlers there can leave residue, so candidacy
//!   excludes them via the shared
//!   [`PeepholeOptimizations::expr_has_specialized_unused_handler`].
//! - class candidacy requires
//!   [`PeepholeOptimizations::classify_class_removability`] (shared with
//!   `remove_unused_class` itself) to return `RemovesClean` — removal must
//!   not bail AND must extract nothing into live code.
//!
//! Side-effect judgments here use a context with strictly less information
//! than the peephole `TraverseCtx` (no tracked constant values). Extra
//! information only ever proves MORE expressions pure, so "pure here" implies
//! "pure at the removal site" — the direction the invariant needs.
//!
//! ## Allocation discipline
//!
//! Everything sized by the program lives in the arena, like
//! `PassDirty::dead_refs`. References are filtered at record time on
//! candidate-kind `SymbolFlags` (most references cost one flag check and no
//! storage), roots are deduplicated at record time by marking them live
//! immediately, and the edge "graph" is a flat list sorted by source symbol,
//! range-scanned via `partition_point` — no per-candidate allocations.

use std::cell::Cell;

use oxc_allocator::{Allocator, BitSet, Vec as ArenaVec};
use oxc_ast::ast::*;
use oxc_ast_visit::{
    Visit,
    walk::{
        walk_class, walk_export_default_declaration, walk_export_named_declaration, walk_function,
        walk_variable_declarator,
    },
};
use oxc_ecmascript::{
    GlobalContext,
    side_effects::{
        MayHaveSideEffects, MayHaveSideEffectsContext, PropertyReadSideEffects, is_pure_function,
    },
};
use oxc_semantic::{IsGlobalReference, Scoping};
use oxc_syntax::{
    reference::ReferenceId,
    scope::{ScopeFlags, ScopeId},
    symbol::{SymbolFlags, SymbolId},
};

use crate::{
    CompressOptions, CompressOptionsUnused,
    peephole::{ClassRemovability, PeepholeOptimizations},
};

/// Symbol kinds a liveness candidate can have: function/class declarations
/// and `var`/`let`/`const` bindings. A necessary (not sufficient) condition
/// for candidacy.
const CANDIDATE_KINDS: SymbolFlags = SymbolFlags::FunctionScopedVariable
    .union(SymbolFlags::BlockScopedVariable)
    .union(SymbolFlags::Class)
    .union(SymbolFlags::Function);

/// Whether the driver must recompute [`compute_dead_symbols`] after a flush:
/// a reference to a candidate-kind symbol was pruned while the symbol still
/// has remaining references — the only transition that can turn a live
/// reference cycle dead. A dropped direct eval also triggers (the flag
/// refresh may clear the root flag that forces [`compute_dead_symbols`] to
/// skip). On large inputs most passes prune such references, so this fires
/// often; its job is to skip the walk for quiet/late iterations, not to make
/// recomputes rare.
///
/// Must be called AFTER `retain_resolved_references_excluding`: the prune
/// only shrinks per-symbol resolved-reference lists, `Reference` entries
/// survive, so the pruned references can still be inspected here.
pub fn recompute_trigger(
    scoping: &Scoping,
    options: &CompressOptions,
    dead_refs: &BitSet<'_>,
    eval_dropped: bool,
) -> bool {
    if options.unused == CompressOptionsUnused::Keep {
        return false;
    }
    if eval_dropped {
        return true;
    }
    if scoping.root_scope_flags().contains_direct_eval() {
        // `compute_dead_symbols` would skip anyway.
        return false;
    }
    dead_refs.ones().any(|idx| {
        scoping.get_reference(ReferenceId::from_usize(idx)).symbol_id().is_some_and(|symbol_id| {
            scoping.symbol_flags(symbol_id).intersects(CANDIDATE_KINDS)
                && !scoping.symbol_is_unused(symbol_id)
        })
    })
}

/// Side-effect context for use outside the peephole traversal. Mirrors the
/// `TraverseCtx` impls in `traverse_context/ecma_context.rs`, minus the
/// tracked-constant lookups (the trait defaults are strictly more
/// conservative).
struct LivenessCtx<'b> {
    scoping: &'b Scoping,
    options: &'b CompressOptions,
}

impl<'a> GlobalContext<'a> for LivenessCtx<'_> {
    fn is_global_reference(&self, ident: &IdentifierReference<'a>) -> bool {
        ident.is_global_reference(self.scoping)
    }
}

impl MayHaveSideEffectsContext<'_> for LivenessCtx<'_> {
    fn annotations(&self) -> bool {
        self.options.treeshake.annotations
    }

    fn manual_pure_functions(&self, callee: &Expression) -> bool {
        is_pure_function(callee, &self.options.treeshake.manual_pure_functions)
    }

    fn property_read_side_effects(&self) -> PropertyReadSideEffects {
        self.options.treeshake.property_read_side_effects
    }

    fn property_write_side_effects(&self) -> bool {
        self.options.treeshake.property_write_side_effects
    }

    fn unknown_global_side_effects(&self) -> bool {
        self.options.treeshake.unknown_global_side_effects
    }
}

/// Compute the set of candidate symbols no live code can reach. Bits are
/// `SymbolId::index()`; ids minted after this runs are beyond capacity and
/// read as live (`BitSet::contains` is `false` past capacity), matching the
/// `PassDirty::dead_refs` convention.
pub fn compute_dead_symbols<'a>(
    program: &Program<'a>,
    scoping: &Scoping,
    options: &CompressOptions,
    allocator: &'a Allocator,
) -> BitSet<'a> {
    let empty = || BitSet::new_in(0, allocator);
    if options.unused == CompressOptionsUnused::Keep {
        return empty();
    }
    // Mirrors `can_remove_unused_declarators` and subsumes the per-site
    // current-scope checks: any direct eval flags the root via ancestor
    // propagation (see `refresh_direct_eval_flags`).
    if scoping.root_scope_flags().contains_direct_eval() {
        return empty();
    }

    let symbols_len = scoping.symbols_len();
    let mut collector = Collector {
        ctx: LivenessCtx { scoping, options },
        is_script: program.source_type.is_script(),
        scope_depth: 0,
        in_export: false,
        enclosing_candidate: None,
        candidates: BitSet::new_in(symbols_len, allocator),
        live: BitSet::new_in(symbols_len, allocator),
        roots: ArenaVec::new_in(&allocator),
        edges: ArenaVec::new_in(&allocator),
    };
    collector.visit_program(program);

    let Collector { candidates, mut live, roots, mut edges, .. } = collector;
    if candidates.is_empty() {
        return empty();
    }

    // The flat edge list sorted by source symbol is the adjacency "map":
    // a live symbol's targets are one `partition_point` range scan away.
    edges.sort_unstable_by_key(|&(from, _)| from.index());

    // Roots are already marked live (record-time dedup); propagate.
    // Marking non-candidate targets live is harmless — only candidates are
    // consulted below.
    let mut worklist = roots;
    while let Some(symbol_id) = worklist.pop() {
        // Only candidates have outgoing edges by construction.
        if !candidates.contains(symbol_id.index()) {
            continue;
        }
        let start = edges.partition_point(|&(from, _)| from.index() < symbol_id.index());
        for &(_, target) in edges[start..].iter().take_while(|&&(from, _)| from == symbol_id) {
            if !live.contains(target.index()) {
                live.set_bit(target.index());
                worklist.push(target);
            }
        }
    }

    let mut dead = BitSet::new_in(symbols_len, allocator);
    for candidate in candidates.ones() {
        if !live.contains(candidate) {
            dead.set_bit(candidate);
        }
    }
    dead
}

struct Collector<'a, 'b> {
    ctx: LivenessCtx<'b>,
    is_script: bool,
    /// Scope nesting depth; 1 = the program (root) scope.
    scope_depth: usize,
    /// The visited node is the `declaration` of an export statement (or a
    /// sibling declarator of one). Cleared for the declaration's subtree so
    /// declarations nested inside an exported function/class stay eligible.
    in_export: bool,
    /// Innermost candidate whose deferred region we are inside.
    enclosing_candidate: Option<SymbolId>,
    candidates: BitSet<'a>,
    /// Marked at record time for roots, during propagation for edge targets.
    live: BitSet<'a>,
    /// Deduplicated live-context references to candidate-kind symbols; the
    /// propagation worklist.
    roots: ArenaVec<'a, SymbolId>,
    /// `(innermost enclosing candidate, referenced candidate-kind symbol)`.
    edges: ArenaVec<'a, (SymbolId, SymbolId)>,
}

impl Collector<'_, '_> {
    /// `keep_top_level_var_in_script_mode` mirror: script-mode statements
    /// visited at the root scope are not removable.
    fn scope_allows_removal(&self) -> bool {
        !(self.is_script && self.scope_depth <= 1)
    }

    /// Shared visitor skeleton for the three declaration kinds: clear
    /// `in_export` for the subtree, track the innermost enclosing candidate
    /// across the walk, restore both.
    fn walk_declaration(&mut self, candidate: Option<SymbolId>, walk: impl FnOnce(&mut Self)) {
        let saved_export = std::mem::replace(&mut self.in_export, false);
        let saved_candidate = self.enclosing_candidate;
        if let Some(symbol_id) = candidate {
            self.candidates.set_bit(symbol_id.index());
            self.enclosing_candidate = Some(symbol_id);
        }
        walk(self);
        self.enclosing_candidate = saved_candidate;
        self.in_export = saved_export;
    }

    /// Init shapes `remove_unused_expression` fully drops when pure. Kinds
    /// with specialized handlers can leave residue (a surviving expression
    /// whose references would dangle once the cycle is removed), so they are
    /// not candidates.
    fn init_fully_removable(&self, init: Option<&Expression<'_>>) -> bool {
        match init {
            None
            | Some(Expression::FunctionExpression(_) | Expression::ArrowFunctionExpression(_)) => {
                true
            }
            Some(e) => {
                !PeepholeOptimizations::expr_has_specialized_unused_handler(e)
                    && !e.may_have_side_effects(&self.ctx)
            }
        }
    }
}

impl<'a> Visit<'a> for Collector<'_, '_> {
    fn enter_scope(&mut self, _flags: ScopeFlags, _scope_id: &Cell<Option<ScopeId>>) {
        self.scope_depth += 1;
    }

    fn leave_scope(&mut self) {
        self.scope_depth -= 1;
    }

    fn visit_identifier_reference(&mut self, it: &IdentifierReference<'a>) {
        if let Some(reference_id) = it.reference_id.get()
            && let Some(symbol_id) = self.ctx.scoping.get_reference(reference_id).symbol_id()
            && self.ctx.scoping.symbol_flags(symbol_id).intersects(CANDIDATE_KINDS)
        {
            match self.enclosing_candidate {
                None => {
                    if !self.live.contains(symbol_id.index()) {
                        self.live.set_bit(symbol_id.index());
                        self.roots.push(symbol_id);
                    }
                }
                Some(from) => self.edges.push((from, symbol_id)),
            }
        }
    }

    fn visit_export_named_declaration(&mut self, it: &ExportNamedDeclaration<'a>) {
        let saved = std::mem::replace(&mut self.in_export, true);
        walk_export_named_declaration(self, it);
        self.in_export = saved;
    }

    fn visit_export_default_declaration(&mut self, it: &ExportDefaultDeclaration<'a>) {
        let saved = std::mem::replace(&mut self.in_export, true);
        walk_export_default_declaration(self, it);
        self.in_export = saved;
    }

    fn visit_function(&mut self, it: &Function<'a>, flags: ScopeFlags) {
        let candidate = if it.is_declaration() && !self.in_export && self.scope_allows_removal() {
            it.id.as_ref().and_then(|id| id.symbol_id.get())
        } else {
            None
        };
        self.walk_declaration(candidate, |v| walk_function(v, it, flags));
    }

    fn visit_class(&mut self, it: &Class<'a>) {
        let candidate = if it.is_declaration()
            && !self.in_export
            && self.scope_allows_removal()
            && matches!(
                PeepholeOptimizations::classify_class_removability(it, &self.ctx),
                ClassRemovability::RemovesClean
            ) {
            it.id.as_ref().and_then(|id| id.symbol_id.get())
        } else {
            None
        };
        self.walk_declaration(candidate, |v| walk_class(v, it));
    }

    fn visit_variable_declarator(&mut self, it: &VariableDeclarator<'a>) {
        let candidate = if !self.in_export
            && self.scope_allows_removal()
            && !it.kind.is_using()
            && self.init_fully_removable(it.init.as_ref())
        {
            match &it.id {
                BindingPattern::BindingIdentifier(id) => id.symbol_id.get(),
                _ => None,
            }
        } else {
            None
        };
        self.walk_declaration(candidate, |v| walk_variable_declarator(v, it));
    }
}
