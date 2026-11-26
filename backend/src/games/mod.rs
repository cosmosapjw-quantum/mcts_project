pub mod gomoku15;
#[cfg(any(test, feature = "mqft-train"))]
pub mod gomoku;
#[cfg(test)]
pub mod tictactoe;
pub use gomoku15::Gomoku15;
#[cfg(test)]
pub use gomoku::Gomoku;
#[cfg(test)]
pub use tictactoe::TicTacToe;
