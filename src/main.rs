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
      if args.is_empty() {
        print_usage();
        return Err("inspect requires <input.pt>".into());
      }
      let input = PathBuf::from(&args[0]);
      let load_opts = parse_load_options(&args[1..])?;
      let checkpoint = PtCheckpoint::load(&input, load_opts)?;
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
      let mut load_opts = LoadOptions::default();
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
          "--state-dict-root-key" => {
            let Some(value) = args.get(index + 1) else {
              return Err("--state-dict-root-key requires a value".into());
            };
            load_opts.state_dict_root_keys.push(value.clone());
            index += 2;
          }
          "--state-dict-root-strict" => {
            let Some(value) = args.get(index + 1) else {
              return Err("--state-dict-root-strict requires a value".into());
            };
            load_opts.state_dict_root_strict = parse_bool_flag(value)
              .ok_or_else(|| "--state-dict-root-strict expects true|false".to_string())?;
            index += 2;
          }
          other => {
            return Err(format!("unknown argument: {}", other).into());
          }
        }
      }

      let checkpoint = PtCheckpoint::load(&input, load_opts)?;
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
  eprintln!("  pt-loader inspect <input.pt> [--state-dict-root-key <key>]... [--state-dict-root-strict <true|false>]");
  eprintln!(
    "  pt-loader convert <input.pt> [--out-dir <dir>] [--state-dict-root-key <key>]... [--state-dict-root-strict <true|false>]"
  );
}

fn parse_bool_flag(value: &str) -> Option<bool> {
  match value {
    "true" | "1" | "yes" | "on" => Some(true),
    "false" | "0" | "no" | "off" => Some(false),
    _ => None,
  }
}

fn parse_load_options(args: &[String]) -> Result<LoadOptions, Box<dyn std::error::Error>> {
  let mut opts = LoadOptions::default();
  let mut index = 0usize;
  while index < args.len() {
    match args[index].as_str() {
      "--state-dict-root-key" => {
        let Some(value) = args.get(index + 1) else {
          return Err("--state-dict-root-key requires a value".into());
        };
        opts.state_dict_root_keys.push(value.clone());
        index += 2;
      }
      "--state-dict-root-strict" => {
        let Some(value) = args.get(index + 1) else {
          return Err("--state-dict-root-strict requires a value".into());
        };
        opts.state_dict_root_strict = parse_bool_flag(value)
          .ok_or_else(|| "--state-dict-root-strict expects true|false".to_string())?;
        index += 2;
      }
      other => return Err(format!("unknown argument: {}", other).into()),
    }
  }
  Ok(opts)
}
