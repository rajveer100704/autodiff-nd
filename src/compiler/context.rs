use std::cell::RefCell;
use crate::compiler::Graph;

thread_local! {
    pub static GRAPH_CTX: RefCell<Option<Graph>> = RefCell::new(None);
}

/// Returns true if the current thread is recording a lazy graph.
pub fn is_tracing() -> bool {
    GRAPH_CTX.with(|ctx| ctx.borrow().is_some())
}

/// Provides access to the current active graph for recording nodes.
pub fn with_graph<F, R>(f: F) -> R 
where 
    F: FnOnce(&mut Graph) -> R 
{
    GRAPH_CTX.with(|ctx| {
        let mut borrow = ctx.borrow_mut();
        let graph = borrow.as_mut().expect("Not in a lazy tracing context");
        f(graph)
    })
}

/// Scoped tracing block. All operations inside this block will be recorded 
/// into a lazy graph instead of being executed immediately.
pub fn lazy<F, R>(f: F) -> (R, Graph) 
where 
    F: FnOnce() -> R 
{
    // Initialize a new graph in the thread local
    GRAPH_CTX.with(|ctx| {
        *ctx.borrow_mut() = Some(Graph::new());
    });

    let result = f();

    // Take the graph out of the thread local
    let graph = GRAPH_CTX.with(|ctx| {
        ctx.borrow_mut().take().expect("Graph vanished during tracing")
    });

    (result, graph)
}
