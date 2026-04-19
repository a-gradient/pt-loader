use pt_loader::{ExportFormat, ExportOptions, LoadOptions, PtCheckpoint};
use std::env;
use std::path::PathBuf;

fn main() {
  if let Err(error) = run() {
    eprintln!("error: {}", error);
    std::process::exit(1);
  }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = env::args().skip(1).collect::<Vec<_>>();
  if args.is_empty() {
    print_usage();
    return Err("missing command".into());
  }

  let command = args.remove(0);
  match command.as_str() {
    "inspect" => {
      if args.len() != 1 {
        print_usage();
        return Err("inspect requires exactly one argument: <input.pt>".into());
      }
      let input = PathBuf::from(&args[0]);
      let checkpoint = PtCheckpoint::load(&input, LoadOptions::default())?;
      println!("{}", serde_json::to_string_pretty(checkpoint.metadata())?);
      Ok(())
    }
    "convert" => {
      if args.is_empty() {
        print_usage();
        return Err("convert requires <input.pt> and optional --out-dir <dir>".into());
      }
      let input = PathBuf::from(&args[0]);
      let mut out_dir = PathBuf::from("out");
      let mut index = 1usize;
      while index < args.len() {
        match args[index].as_str() {
          "--out-dir" => {
            let Some(value) = args.get(index + 1) else {
              return Err("--out-dir requires a value".into());
            };
            out_dir = PathBuf::from(value);
            index += 2;
          }
          other => {
            return Err(format!("unknown argument: {}", other).into());
          }
        }
      }

      let checkpoint = PtCheckpoint::load(&input, LoadOptions::default())?;
      let result = checkpoint.export(&out_dir, ExportOptions::new(ExportFormat::Safetensors, Some(&input)))?;
      println!("{}", serde_json::to_string_pretty(&result)?);
      Ok(())
    }
    _ => {
      print_usage();
      Err(format!("unknown command: {}", command).into())
    }
  }
}

fn print_usage() {
  eprintln!("Usage:");
  eprintln!("  pt-loader inspect <input.pt>");
  eprintln!("  pt-loader convert <input.pt> [--out-dir <dir>]");
}
