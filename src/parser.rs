use std::collections::HashMap;

use crate::types::{ConvertError, DType, LoadOptions, NumpyScalarData, Result, StorageRef, TensorRef, Value};

pub(crate) fn parse_pickle(input: &[u8], opts: &LoadOptions) -> Result<Value> {
  PickleParser::new(input, opts).parse()
}

struct PickleParser<'a> {
  input: &'a [u8],
  pos: usize,
  stack: Vec<Value>,
  memo: HashMap<usize, Value>,
  next_memo: usize,
  opts: &'a LoadOptions,
}

impl<'a> PickleParser<'a> {
  fn new(input: &'a [u8], opts: &'a LoadOptions) -> Self {
    Self {
      input,
      pos: 0,
      stack: Vec::new(),
      memo: HashMap::new(),
      next_memo: 0,
      opts,
    }
  }

  fn parse(mut self) -> Result<Value> {
    while self.pos < self.input.len() {
      let offset = self.pos;
      let opcode = self.read_u8()?;
      match opcode {
        b'\x80' => {
          let protocol = self.read_u8()?;
          if protocol < 2 {
            return Err(ConvertError::UnsupportedFormat(format!(
              "pickle protocol {} is not supported",
              protocol
            )));
          }
        }
        b'\x95' => {
          let _frame_len = self.read_u64_le()?;
        }
        b'.' => {
          let result = self
            .stack
            .pop()
            .ok_or_else(|| ConvertError::InvalidStructure("empty stack at STOP".to_string()))?;
          return Ok(result);
        }
        b'(' => self.stack.push(Value::Marker),
        b'}' => self.stack.push(Value::Dict(Vec::new())),
        b']' => self.stack.push(Value::List(Vec::new())),
        b')' => self.stack.push(Value::Tuple(Vec::new())),
        b'N' => self.stack.push(Value::None),
        b'\x88' => self.stack.push(Value::Bool(true)),
        b'\x89' => self.stack.push(Value::Bool(false)),
        b'K' => {
          let value = self.read_u8()? as i64;
          self.stack.push(Value::Int(value));
        }
        b'M' => {
          let value = self.read_u16_le()? as i64;
          self.stack.push(Value::Int(value));
        }
        b'J' => {
          let value = self.read_i32_le()? as i64;
          self.stack.push(Value::Int(value));
        }
        b'\x8a' => {
          let value = self.read_long1()?;
          self.stack.push(Value::Int(value));
        }
        b'\x8b' => {
          let value = self.read_long4()?;
          self.stack.push(Value::Int(value));
        }
        b'G' => {
          let bits = self.read_u64_be()?;
          self.stack.push(Value::Float(f64::from_bits(bits)));
        }
        b'X' => {
          let len = self.read_u32_le()? as usize;
          let s = self.read_string(len)?;
          self.stack.push(Value::String(s));
        }
        b'\x8c' => {
          let len = self.read_u8()? as usize;
          let s = self.read_string(len)?;
          self.stack.push(Value::String(s));
        }
        b'B' => {
          let len = self.read_u32_le()? as usize;
          let value = self.read_bytes(len)?.to_vec();
          self.stack.push(Value::Bytes(value));
        }
        b'U' => {
          let len = self.read_u8()? as usize;
          let value = self.read_bytes(len)?.to_vec();
          self.stack.push(Value::Bytes(value));
        }
        b'c' => {
          let module = self.read_line()?;
          let name = self.read_line()?;
          self.stack.push(Value::Global { module, name });
        }
        b'h' => {
          let idx = self.read_u8()? as usize;
          let value = self
            .memo
            .get(&idx)
            .cloned()
            .ok_or_else(|| ConvertError::InvalidStructure(format!("missing memo entry {}", idx)))?;
          self.stack.push(value);
        }
        b'j' => {
          let idx = self.read_u32_le()? as usize;
          let value = self
            .memo
            .get(&idx)
            .cloned()
            .ok_or_else(|| ConvertError::InvalidStructure(format!("missing memo entry {}", idx)))?;
          self.stack.push(value);
        }
        b'q' => {
          let idx = self.read_u8()? as usize;
          let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| ConvertError::InvalidStructure("BINPUT on empty stack".to_string()))?;
          self.memo.insert(idx, top);
        }
        b'r' => {
          let idx = self.read_u32_le()? as usize;
          let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| ConvertError::InvalidStructure("LONG_BINPUT on empty stack".to_string()))?;
          self.memo.insert(idx, top);
        }
        b'\x94' => {
          let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| ConvertError::InvalidStructure("MEMOIZE on empty stack".to_string()))?;
          let idx = self.next_memo;
          self.next_memo += 1;
          self.memo.insert(idx, top);
        }
        b't' => {
          let items = self.pop_mark_items()?;
          self.stack.push(Value::Tuple(items));
        }
        b'\x85' => {
          let a = self.pop_stack()?;
          self.stack.push(Value::Tuple(vec![a]));
        }
        b'\x86' => {
          let b = self.pop_stack()?;
          let a = self.pop_stack()?;
          self.stack.push(Value::Tuple(vec![a, b]));
        }
        b'\x87' => {
          let c = self.pop_stack()?;
          let b = self.pop_stack()?;
          let a = self.pop_stack()?;
          self.stack.push(Value::Tuple(vec![a, b, c]));
        }
        b's' => {
          let value = self.pop_stack()?;
          let key = self.pop_stack()?;
          let target = self.pop_stack()?;
          let updated = self.apply_setitem(target, key, value)?;
          self.stack.push(updated);
        }
        b'u' => {
          let items = self.pop_mark_items()?;
          let target = self.pop_stack()?;
          let updated = self.apply_setitems(target, items)?;
          self.stack.push(updated);
        }
        b'e' => {
          let items = self.pop_mark_items()?;
          let target = self.pop_stack()?;
          let updated = self.apply_appends(target, items)?;
          self.stack.push(updated);
        }
        b'a' => {
          let value = self.pop_stack()?;
          let target = self.pop_stack()?;
          let updated = self.apply_append(target, value)?;
          self.stack.push(updated);
        }
        b'R' => {
          let args = self.pop_stack()?;
          let func = self.pop_stack()?;
          let reduced = self.apply_reduce(func, args)?;
          self.stack.push(reduced);
        }
        b'\x81' => {
          let args = self.pop_stack()?;
          let cls = self.pop_stack()?;
          let instance = self.apply_newobj(cls, args)?;
          self.stack.push(instance);
        }
        b'b' => {
          let state = self.pop_stack()?;
          let inst = self.pop_stack()?;
          let built = self.apply_build(inst, state)?;
          self.stack.push(built);
        }
        b'Q' => {
          let pid = self.pop_stack()?;
          let storage = self.apply_persistent_id(pid)?;
          self.stack.push(storage);
        }
        _ => {
          return Err(ConvertError::UnsafeOpcode { opcode, offset });
        }
      }

      if self.stack.len() > self.opts.max_tensor_count.saturating_mul(16) {
        return Err(ConvertError::ResourceLimitExceeded(
          "pickle stack grew beyond safe bound".to_string(),
        ));
      }
    }

    Err(ConvertError::InvalidStructure(
      "pickle stream ended without STOP".to_string(),
    ))
  }

  fn apply_setitem(&self, target: Value, key: Value, value: Value) -> Result<Value> {
    match target {
      Value::Dict(mut entries) => {
        entries.push((key, value));
        Ok(Value::Dict(entries))
      }
      Value::OrderedDict(mut entries) => {
        let key = key.as_string()?;
        entries.push((key, value));
        Ok(Value::OrderedDict(entries))
      }
      _ => Err(ConvertError::InvalidStructure(
        "SETITEM target is not dict-like".to_string(),
      )),
    }
  }

  fn apply_setitems(&self, target: Value, items: Vec<Value>) -> Result<Value> {
    if items.len() % 2 != 0 {
      return Err(ConvertError::InvalidStructure(
        "SETITEMS expects even number of stack items".to_string(),
      ));
    }

    match target {
      Value::Dict(mut entries) => {
        for pair in items.chunks_exact(2) {
          entries.push((pair[0].clone(), pair[1].clone()));
        }
        Ok(Value::Dict(entries))
      }
      Value::OrderedDict(mut entries) => {
        for pair in items.chunks_exact(2) {
          let key = pair[0].as_string()?;
          entries.push((key, pair[1].clone()));
        }
        Ok(Value::OrderedDict(entries))
      }
      _ => Err(ConvertError::InvalidStructure(
        "SETITEMS target is not dict-like".to_string(),
      )),
    }
  }

  fn apply_appends(&self, target: Value, items: Vec<Value>) -> Result<Value> {
    match target {
      Value::List(mut list) => {
        list.extend(items);
        Ok(Value::List(list))
      }
      _ => Err(ConvertError::InvalidStructure("APPENDS target is not list".to_string())),
    }
  }

  fn apply_append(&self, target: Value, value: Value) -> Result<Value> {
    match target {
      Value::List(mut list) => {
        list.push(value);
        Ok(Value::List(list))
      }
      _ => Err(ConvertError::InvalidStructure("APPEND target is not list".to_string())),
    }
  }

  fn apply_reduce(&self, func: Value, args: Value) -> Result<Value> {
    let (module, name) = match func {
      Value::Global { module, name } => (module, name),
      _ => {
        return Err(ConvertError::InvalidStructure(
          "REDUCE function is not GLOBAL".to_string(),
        ))
      }
    };

    let args = match args {
      Value::Tuple(items) => items,
      _ => {
        return Err(ConvertError::InvalidStructure(
          "REDUCE args must be a tuple".to_string(),
        ))
      }
    };

    match (module.as_str(), name.as_str()) {
      ("collections", "OrderedDict") => {
        if args.is_empty() {
          return Ok(Value::OrderedDict(Vec::new()));
        }
        if args.len() != 1 {
          return Err(ConvertError::InvalidStructure(
            "OrderedDict reduce expects 0 or 1 argument".to_string(),
          ));
        }
        let items = match &args[0] {
          Value::List(values) | Value::Tuple(values) => values,
          _ => {
            return Err(ConvertError::InvalidStructure(
              "OrderedDict items must be list/tuple of pairs".to_string(),
            ))
          }
        };
        let mut out = Vec::new();
        for item in items {
          let pair = match item {
            Value::Tuple(parts) if parts.len() == 2 => parts,
            _ => {
              return Err(ConvertError::InvalidStructure(
                "OrderedDict item must be tuple(key,value)".to_string(),
              ))
            }
          };
          out.push((pair[0].as_string()?, pair[1].clone()));
        }
        Ok(Value::OrderedDict(out))
      }
      ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
        if args.len() < 4 {
          return Err(ConvertError::InvalidStructure(
            "_rebuild_tensor requires at least 4 args".to_string(),
          ));
        }
        let storage = match &args[0] {
          Value::StorageRef(value) => value.clone(),
          _ => {
            return Err(ConvertError::InvalidStructure(
              "tensor rebuild arg0 must be storage ref".to_string(),
            ))
          }
        };
        let offset_elems = args[1].as_usize()?;
        let shape = args[2].as_usize_vec()?;
        let stride = args[3].as_usize_vec()?;
        Ok(Value::TensorRef(TensorRef {
          storage,
          offset_elems,
          shape,
          stride,
        }))
      }
      ("torch._utils", "_rebuild_parameter") => {
        if args.is_empty() {
          return Err(ConvertError::InvalidStructure(
            "_rebuild_parameter needs tensor arg".to_string(),
          ));
        }
        Ok(args[0].clone())
      }
      ("__builtin__", "set") | ("builtins", "set") | ("__builtin__", "frozenset") | ("builtins", "frozenset") => {
        if args.is_empty() {
          return Ok(Value::Set(Vec::new()));
        }
        if args.len() != 1 {
          return Err(ConvertError::InvalidStructure(
            "set reduce expects 0 or 1 argument".to_string(),
          ));
        }
        let items = match &args[0] {
          Value::List(items) | Value::Tuple(items) | Value::Set(items) => items.clone(),
          _ => {
            return Err(ConvertError::InvalidStructure(
              "set reduce argument must be list/tuple/set".to_string(),
            ))
          }
        };
        Ok(Value::Set(items))
      }
      ("torch", "Size") => {
        if args.len() != 1 {
          return Err(ConvertError::InvalidStructure(
            "torch.Size reduce expects one argument".to_string(),
          ));
        }
        match &args[0] {
          Value::Tuple(items) | Value::List(items) => Ok(Value::Tuple(items.clone())),
          other => Err(ConvertError::InvalidStructure(format!(
            "torch.Size argument must be tuple/list, got {:?}",
            other
          ))),
        }
      }
      ("numpy._core.multiarray", "scalar") | ("numpy.core.multiarray", "scalar") => {
        if let Some(value) = recover_numpy_scalar_call(&args) {
          Ok(value)
        } else {
          Ok(Value::Call {
            func: format!("{module}.{name}"),
            args,
            state: None,
          })
        }
      }
      _ => Ok(Value::Call {
        func: format!("{module}.{name}"),
        args,
        state: None,
      }),
    }
  }

  fn apply_newobj(&self, cls: Value, args: Value) -> Result<Value> {
    let (module, name) = match cls {
      Value::Global { module, name } => (module, name),
      _ => return Err(ConvertError::InvalidStructure("NEWOBJ class is not GLOBAL".to_string())),
    };
    let _args = match args {
      Value::Tuple(items) => items,
      _ => return Err(ConvertError::InvalidStructure("NEWOBJ args must be tuple".to_string())),
    };

    match (module.as_str(), name.as_str()) {
      ("collections", "OrderedDict") => Ok(Value::OrderedDict(Vec::new())),
      _ => Ok(Value::Object {
        module,
        name,
        args: _args,
        state: None,
      }),
    }
  }

  fn apply_build(&self, inst: Value, state: Value) -> Result<Value> {
    match (inst, state) {
      (Value::OrderedDict(entries), Value::None) => Ok(Value::OrderedDict(entries)),
      (Value::TensorRef(spec), Value::None) => Ok(Value::TensorRef(spec)),
      (Value::Object { module, name, args, .. }, state) => Ok(Value::Object {
        module,
        name,
        args,
        state: Some(Box::new(state)),
      }),
      (Value::Call { func, args, .. }, state) => Ok(Value::Call {
        func,
        args,
        state: Some(Box::new(state)),
      }),
      (_, Value::None) => Err(ConvertError::InvalidStructure(
        "BUILD unsupported for this object type".to_string(),
      )),
      _ => Err(ConvertError::InvalidStructure(
        "BUILD with non-empty state is not supported".to_string(),
      )),
    }
  }

  fn apply_persistent_id(&self, pid: Value) -> Result<Value> {
    let parts = match pid {
      Value::Tuple(parts) => parts,
      _ => {
        return Err(ConvertError::InvalidStructure(
          "BINPERSID expects tuple persistent id".to_string(),
        ))
      }
    };
    if parts.len() < 5 {
      return Err(ConvertError::InvalidStructure(
        "persistent id tuple is too short".to_string(),
      ));
    }

    let kind = parts[0].as_string()?;
    if kind != "storage" {
      return Err(ConvertError::InvalidStructure(format!(
        "unsupported persistent id kind: {}",
        kind
      )));
    }

    let dtype = match &parts[1] {
      Value::Global { module: _, name } => storage_name_to_dtype(name)?,
      Value::String(name) => storage_name_to_dtype(name)?,
      other => {
        return Err(ConvertError::InvalidStructure(format!(
          "unexpected storage dtype tag: {:?}",
          other
        )))
      }
    };
    let key = parts[2].as_string()?;
    let size_elems = parts[4].as_usize()?;

    Ok(Value::StorageRef(StorageRef { key, dtype, size_elems }))
  }

  fn pop_mark_items(&mut self) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    while let Some(value) = self.stack.pop() {
      if matches!(value, Value::Marker) {
        out.reverse();
        return Ok(out);
      }
      out.push(value);
    }
    Err(ConvertError::InvalidStructure("MARK not found on stack".to_string()))
  }

  fn pop_stack(&mut self) -> Result<Value> {
    self
      .stack
      .pop()
      .ok_or_else(|| ConvertError::InvalidStructure("stack underflow".to_string()))
  }

  fn read_u8(&mut self) -> Result<u8> {
    if self.pos >= self.input.len() {
      return Err(ConvertError::InvalidStructure("unexpected EOF".to_string()));
    }
    let byte = self.input[self.pos];
    self.pos += 1;
    Ok(byte)
  }

  fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
    if self.pos + len > self.input.len() {
      return Err(ConvertError::InvalidStructure("unexpected EOF".to_string()));
    }
    let out = &self.input[self.pos..self.pos + len];
    self.pos += len;
    Ok(out)
  }

  fn read_u16_le(&mut self) -> Result<u16> {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(self.read_bytes(2)?);
    Ok(u16::from_le_bytes(buf))
  }

  fn read_u32_le(&mut self) -> Result<u32> {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(self.read_bytes(4)?);
    Ok(u32::from_le_bytes(buf))
  }

  fn read_i32_le(&mut self) -> Result<i32> {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(self.read_bytes(4)?);
    Ok(i32::from_le_bytes(buf))
  }

  fn read_u64_le(&mut self) -> Result<u64> {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(self.read_bytes(8)?);
    Ok(u64::from_le_bytes(buf))
  }

  fn read_u64_be(&mut self) -> Result<u64> {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(self.read_bytes(8)?);
    Ok(u64::from_be_bytes(buf))
  }

  fn read_line(&mut self) -> Result<String> {
    let start = self.pos;
    while self.pos < self.input.len() {
      if self.input[self.pos] == b'\n' {
        let line = std::str::from_utf8(&self.input[start..self.pos])
          .map_err(|_| ConvertError::InvalidStructure("GLOBAL line is not utf-8".to_string()))?;
        self.pos += 1;
        return Ok(line.to_string());
      }
      self.pos += 1;
    }
    Err(ConvertError::InvalidStructure("unterminated GLOBAL line".to_string()))
  }

  fn read_string(&mut self, len: usize) -> Result<String> {
    let bytes = self.read_bytes(len)?;
    let value =
      std::str::from_utf8(bytes).map_err(|_| ConvertError::InvalidStructure("invalid utf-8 string".to_string()))?;
    Ok(value.to_string())
  }

  fn read_long1(&mut self) -> Result<i64> {
    let len = self.read_u8()? as usize;
    self.read_long_bytes(len)
  }

  fn read_long4(&mut self) -> Result<i64> {
    let len = self.read_u32_le()? as usize;
    self.read_long_bytes(len)
  }

  fn read_long_bytes(&mut self, len: usize) -> Result<i64> {
    if len == 0 {
      return Ok(0);
    }
    if len > 8 {
      return Err(ConvertError::InvalidStructure(
        "LONG integer wider than 64-bit is not supported".to_string(),
      ));
    }
    let bytes = self.read_bytes(len)?;

    let mut value: i128 = 0;
    for (idx, byte) in bytes.iter().enumerate() {
      value |= (*byte as i128) << (idx * 8);
    }

    if bytes[len - 1] & 0x80 != 0 {
      value -= 1_i128 << (len * 8);
    }

    i64::try_from(value).map_err(|_| ConvertError::InvalidStructure("LONG integer does not fit in i64".to_string()))
  }
}

fn storage_name_to_dtype(name: &str) -> Result<DType> {
  match name {
    "FloatStorage" => Ok(DType::F32),
    "DoubleStorage" => Ok(DType::F64),
    "HalfStorage" => Ok(DType::F16),
    "BFloat16Storage" => Ok(DType::BF16),
    "ByteStorage" => Ok(DType::U8),
    "CharStorage" => Ok(DType::I8),
    "ShortStorage" => Ok(DType::I16),
    "IntStorage" => Ok(DType::I32),
    "LongStorage" => Ok(DType::I64),
    "BoolStorage" => Ok(DType::Bool),
    _ => Err(ConvertError::InvalidStructure(format!(
      "unsupported torch storage type {}",
      name
    ))),
  }
}

#[derive(Debug, Clone, Copy)]
enum NumpyEndian {
  Little,
  Big,
  NotApplicable,
}

fn recover_numpy_scalar_call(args: &[Value]) -> Option<Value> {
  if args.len() != 2 {
    return None;
  }
  let (dtype, endian) = recover_numpy_dtype(&args[0])?;
  let raw = recover_numpy_scalar_bytes(&args[1])?;
  let data = decode_numpy_scalar(dtype, endian, &raw)?;
  Some(Value::NumpyScalar {
    dtype,
    shape: Vec::new(),
    data,
  })
}

fn recover_numpy_dtype(value: &Value) -> Option<(DType, NumpyEndian)> {
  let Value::Call { func, args, state } = value else {
    return None;
  };
  if func != "numpy.dtype" {
    return None;
  }
  let code = args.first()?.as_string().ok()?;

  let mut code_bytes = code.as_bytes();
  let mut byte_order = None;
  if let Some(prefix) = code_bytes.first().copied() {
    if matches!(prefix, b'<' | b'>' | b'|' | b'=') {
      byte_order = Some(prefix as char);
      code_bytes = &code_bytes[1..];
    }
  }

  if byte_order.is_none() {
    if let Some(state) = state {
      if let Value::Tuple(items) | Value::List(items) = state.as_ref() {
        if let Some(Value::String(order)) = items.get(1) {
          byte_order = order.chars().next();
        }
      }
    }
  }

  let dtype = map_numpy_dtype_code(std::str::from_utf8(code_bytes).ok()?)?;
  let endian = match byte_order? {
    '<' => NumpyEndian::Little,
    '>' => NumpyEndian::Big,
    '|' => NumpyEndian::NotApplicable,
    '=' => return None,
    _ => return None,
  };

  if matches!(endian, NumpyEndian::NotApplicable) && dtype.elem_size() > 1 {
    return None;
  }

  Some((dtype, endian))
}

fn map_numpy_dtype_code(code: &str) -> Option<DType> {
  match code {
    "f2" => Some(DType::F16),
    "f4" => Some(DType::F32),
    "f8" => Some(DType::F64),
    "i1" => Some(DType::I8),
    "i2" => Some(DType::I16),
    "i4" => Some(DType::I32),
    "i8" => Some(DType::I64),
    "u1" => Some(DType::U8),
    "b1" => Some(DType::Bool),
    "bf16" | "bfloat16" => Some(DType::BF16),
    _ => None,
  }
}

fn recover_numpy_scalar_bytes(value: &Value) -> Option<Vec<u8>> {
  match value {
    Value::Bytes(bytes) => Some(bytes.clone()),
    Value::Call { func, args, state } if (func == "_codecs.encode" || func == "codecs.encode") && state.is_none() => {
      if args.len() != 2 {
        return None;
      }
      let text = match &args[0] {
        Value::String(s) => s,
        _ => return None,
      };
      let encoding = match &args[1] {
        Value::String(s) => s,
        _ => return None,
      };
      if encoding != "latin1" {
        return None;
      }
      text
        .chars()
        .map(|ch| u8::try_from(ch as u32).ok())
        .collect::<Option<Vec<u8>>>()
    }
    _ => None,
  }
}

fn decode_numpy_scalar(dtype: DType, endian: NumpyEndian, raw: &[u8]) -> Option<NumpyScalarData> {
  if raw.len() != dtype.elem_size() {
    return None;
  }
  Some(match dtype {
    DType::F16 => {
      let bits = read_u16(raw, endian)?;
      NumpyScalarData::F32(f16_bits_to_f32(bits))
    }
    DType::BF16 => {
      let bits = read_u16(raw, endian)?;
      NumpyScalarData::F32(f32::from_bits((bits as u32) << 16))
    }
    DType::F32 => {
      let bits = read_u32(raw, endian)?;
      NumpyScalarData::F32(f32::from_bits(bits))
    }
    DType::F64 => {
      let bits = read_u64(raw, endian)?;
      NumpyScalarData::F64(f64::from_bits(bits))
    }
    DType::I8 => NumpyScalarData::I8(raw[0] as i8),
    DType::I16 => NumpyScalarData::I16(read_u16(raw, endian)? as i16),
    DType::I32 => NumpyScalarData::I32(read_u32(raw, endian)? as i32),
    DType::I64 => NumpyScalarData::I64(read_u64(raw, endian)? as i64),
    DType::U8 => NumpyScalarData::U8(raw[0]),
    DType::Bool => NumpyScalarData::Bool(raw[0] != 0),
  })
}

fn read_u16(raw: &[u8], endian: NumpyEndian) -> Option<u16> {
  let bytes: [u8; 2] = raw.try_into().ok()?;
  Some(match endian {
    NumpyEndian::Little => u16::from_le_bytes(bytes),
    NumpyEndian::Big => u16::from_be_bytes(bytes),
    NumpyEndian::NotApplicable => u16::from_le_bytes(bytes),
  })
}

fn read_u32(raw: &[u8], endian: NumpyEndian) -> Option<u32> {
  let bytes: [u8; 4] = raw.try_into().ok()?;
  Some(match endian {
    NumpyEndian::Little => u32::from_le_bytes(bytes),
    NumpyEndian::Big => u32::from_be_bytes(bytes),
    NumpyEndian::NotApplicable => u32::from_le_bytes(bytes),
  })
}

fn read_u64(raw: &[u8], endian: NumpyEndian) -> Option<u64> {
  let bytes: [u8; 8] = raw.try_into().ok()?;
  Some(match endian {
    NumpyEndian::Little => u64::from_le_bytes(bytes),
    NumpyEndian::Big => u64::from_be_bytes(bytes),
    NumpyEndian::NotApplicable => u64::from_le_bytes(bytes),
  })
}

fn f16_bits_to_f32(bits: u16) -> f32 {
  let sign = ((bits >> 15) & 0x1) as u32;
  let exp = ((bits >> 10) & 0x1f) as u32;
  let frac = (bits & 0x03ff) as u32;

  let f32_bits = if exp == 0 {
    if frac == 0 {
      sign << 31
    } else {
      let mut mant = frac;
      let mut e = -14i32;
      while (mant & 0x0400) == 0 {
        mant <<= 1;
        e -= 1;
      }
      mant &= 0x03ff;
      let exp32 = (e + 127) as u32;
      (sign << 31) | (exp32 << 23) | (mant << 13)
    }
  } else if exp == 0x1f {
    (sign << 31) | (0xff << 23) | (frac << 13)
  } else {
    let exp32 = (exp as i32 - 15 + 127) as u32;
    (sign << 31) | (exp32 << 23) | (frac << 13)
  };

  f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn numpy_scalar_global() -> Value {
    Value::Global {
      module: "numpy._core.multiarray".to_string(),
      name: "scalar".to_string(),
    }
  }

  fn dtype_call(code: &str, order: &str) -> Value {
    Value::Call {
      func: "numpy.dtype".to_string(),
      args: vec![Value::String(code.to_string()), Value::Bool(false), Value::Bool(true)],
      state: Some(Box::new(Value::Tuple(vec![
        Value::Int(3),
        Value::String(order.to_string()),
        Value::None,
        Value::None,
        Value::None,
        Value::Int(-1),
        Value::Int(-1),
        Value::Int(0),
      ]))),
    }
  }

  fn latin1_codec_bytes(bytes: &[u8]) -> Value {
    let text = bytes.iter().map(|byte| char::from(*byte)).collect::<String>();
    Value::Call {
      func: "_codecs.encode".to_string(),
      args: vec![Value::String(text), Value::String("latin1".to_string())],
      state: None,
    }
  }

  fn parser<'a>(opts: &'a LoadOptions) -> PickleParser<'a> {
    PickleParser::new(&[], opts)
  }

  #[test]
  fn recovers_numpy_scalars() {
    let opts = LoadOptions::default();
    let parser = parser(&opts);
    let cases = vec![
      (
        dtype_call("f4", "<"),
        Value::Bytes(3.5f32.to_le_bytes().to_vec()),
        NumpyScalarData::F32(3.5f32),
      ),
      (
        dtype_call("f8", "<"),
        Value::Bytes((-2.25f64).to_le_bytes().to_vec()),
        NumpyScalarData::F64(-2.25f64),
      ),
      (
        dtype_call("i8", "<"),
        latin1_codec_bytes(&(-7i64).to_le_bytes()),
        NumpyScalarData::I64(-7),
      ),
      (
        dtype_call("u1", "|"),
        latin1_codec_bytes(&[255]),
        NumpyScalarData::U8(255),
      ),
      (
        dtype_call("b1", "|"),
        Value::Bytes(vec![1]),
        NumpyScalarData::Bool(true),
      ),
    ];

    for (dtype, data, expected) in cases {
      let out = parser
        .apply_reduce(numpy_scalar_global(), Value::Tuple(vec![dtype, data]))
        .expect("numpy scalar reduce should parse");
      match out {
        Value::NumpyScalar { dtype: _, shape, data } => {
          assert!(shape.is_empty());
          assert_eq!(data, expected);
        }
        other => panic!("expected numpy scalar recovery, got {:?}", other),
      }
    }
  }

  #[test]
  fn leaves_unrecognized_numpy_scalar_calls_untouched() {
    let opts = LoadOptions::default();
    let parser = parser(&opts);

    let unknown_dtype = parser
      .apply_reduce(
        numpy_scalar_global(),
        Value::Tuple(vec![dtype_call("x9", "<"), Value::Bytes(vec![1, 2, 3, 4])]),
      )
      .expect("unknown dtype should keep call");
    match unknown_dtype {
      Value::Call { func, args, state } => {
        assert_eq!(func, "numpy._core.multiarray.scalar");
        assert_eq!(args.len(), 2);
        assert!(state.is_none());
      }
      other => panic!("expected fallback call, got {:?}", other),
    }

    let native_endian = parser
      .apply_reduce(
        numpy_scalar_global(),
        Value::Tuple(vec![dtype_call("f4", "="), Value::Bytes(1.0f32.to_le_bytes().to_vec())]),
      )
      .expect("native-endian should keep call");
    assert!(matches!(native_endian, Value::Call { .. }));

    let malformed = parser
      .apply_reduce(numpy_scalar_global(), Value::Tuple(vec![dtype_call("f4", "<")]))
      .expect("malformed scalar args should keep call");
    assert!(matches!(malformed, Value::Call { .. }));
  }
}
