use clap::Parser;
use cscp_connector::CscpSharedMemory;
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about = "Control-System-Context-Protocol Standalone Daemon")]
struct Args {
    #[arg(short, long, default_value = "cscp_shm_default")]
    name: String,

    #[arg(short, long, default_value_t = 100)]
    interval_ms: u64,
}

fn main() {
    let args = Args::parse();
    println!(
        "[CSCP Daemon] Starting Control-System-Context-Protocol daemon (Segment: {})",
        args.name
    );

    let shm = CscpSharedMemory::new();
    println!(
        "[CSCP Daemon] SHM Initialized. Magic: 0x{:X}, ABI: 0x{:X}",
        shm.header.magic_signature, shm.header.abi_version
    );

    let mut running = true;
    let mut step_count = 0u64;

    while running {
        thread::sleep(Duration::from_millis(args.interval_ms));
        step_count += 1;

        if step_count % 50 == 0 {
            println!(
                "[CSCP Daemon Status] Heartbeat #{} | Sequence: {}",
                step_count,
                shm.get_sequence()
            );
        }

        if step_count >= 200 {
            println!("[CSCP Daemon] Completed lifecycle run.");
            running = false;
        }
    }
}
