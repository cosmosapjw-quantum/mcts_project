fn main() {
    if let Err(e) = mcts_demo::mqft_train::main_cli() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
