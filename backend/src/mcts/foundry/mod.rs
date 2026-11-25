//! QUARTZ Idea Foundry experimental contracts.
//!
//! The parent module exposes this tree only under Cargo feature
//! `idea-foundry`.  Enabling the feature compiles contracts and tests; it does
//! not install a coordinator into the production MCTS loop.  Experiment code
//! must explicitly construct `FoundryCoordinator` and a guarded executor.

pub mod control;
pub mod coordinator;
pub mod policy;
pub mod search;
pub mod systems;
pub mod types;

pub use control::*;
pub use coordinator::*;
pub use policy::*;
pub use search::*;
pub use systems::*;
pub use types::*;
