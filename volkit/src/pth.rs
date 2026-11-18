//! A torch-free reader for PyTorch `.pth` checkpoints, for files downloaded from untrusted sources
//!
//! A `.pth` is a zip holding `<archive>/data.pkl` (the pickled payload), `<archive>/data/<key>` (one raw storage
//! per key) and `<archive>/byteorder`. The zip reader takes stored (uncompressed) entries only, which is all
//! `torch.save` writes. The pickle runs in a restricted machine that builds plain values (None, bool, int, float,
//! str, bytes, list, tuple, dict, `OrderedDict`) and tensors through `torch._utils._rebuild_tensor_v2` and
//! `_rebuild_parameter`; every other class or function the file names becomes an inert [`Object`] that records
//! the arguments and state the file gives it. Nothing the file names is imported, executed or called
//!
//! Every tensor is checked against its storage when it is built: float32 storages only, a storage exactly as
//! long as its zip entry, contiguous strides and shape × element size within the storage. Little-endian
//! checkpoints only

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::ops::Range;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use memmap2::Mmap;

/// The largest `data.pkl` read: the payload holds names and small values, the tensors live in their own entries
const MAX_PICKLE_BYTES: usize = 32 << 20;
/// The most objects, stack entries or memo entries one pickle may make, which bounds the machine's memory
const MAX_OBJECTS: usize = 1 << 22;
/// The deepest nesting turned into a [`Value`]; a pickle can build cycles, which this stops
const MAX_DEPTH: usize = 128;
/// The most values materialised from the pickle's object graph, which shared references could otherwise blow up
/// exponentially (each level referring to the one below twice)
const MAX_VALUES: usize = 1 << 21;
/// The most dimensions a tensor may have
const MAX_NDIM: usize = 16;

/// The element type of a storage; float32 is the only one the readers built on this need
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// torch `FloatStorage`, little-endian float32
    F32,
}

impl Dtype {
    /// Bytes per element
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
        }
    }
}

/// A tensor in the checkpoint: a contiguous view of one storage, checked against that storage's zip entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    storage: String,
    dtype: Dtype,
    offset: usize,
    shape: Vec<usize>,
}

impl TensorRef {
    /// The storage key (`<archive>/data/<key>`); tensors with the same key and offset share memory in torch
    pub fn storage(&self) -> &str {
        &self.storage
    }

    /// The first element's index in the storage
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The dimensions, outermost first
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The storage's element type
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Elements in the tensor; shapes were checked for overflow when the tensor was built
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// An object of a class outside the allowlist, kept inert: what the file passed to it and stored in it
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    /// The class or callable that would have built it: a [`Value::Global`], or another [`Object`] called
    pub callee: Value,
    /// The arguments of `REDUCE` or `NEWOBJ`
    pub args: Value,
    /// The `BUILD` state, if any
    pub state: Option<Value>,
    /// Values appended to it
    pub items: Vec<Value>,
    /// Key-value pairs set on it
    pub entries: Vec<(Value, Value)>,
}

/// A value from the pickled payload
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Python `None`
    None,
    /// A `bool`
    Bool(bool),
    /// An integer that fits in i64
    Int(i64),
    /// An integer beyond i64, as little-endian two's complement bytes
    BigInt(Vec<u8>),
    /// A `float`
    Float(f64),
    /// A `str`
    Str(String),
    /// A `bytes` or `bytearray`
    Bytes(Vec<u8>),
    /// A `list`
    List(Vec<Value>),
    /// A `tuple`
    Tuple(Vec<Value>),
    /// A dict or `OrderedDict`, in insertion order
    Dict(Vec<(Value, Value)>),
    /// A tensor rebuilt by `torch._utils._rebuild_tensor_v2` (or `_rebuild_parameter`), not yet read
    Tensor(TensorRef),
    /// A class or function the file names outside the allowlist (`module.name`), never imported
    Global(String),
    /// An inert instance of a class outside the allowlist
    Object(Box<Object>),
}

impl Value {
    /// The entries of a dict
    pub fn as_dict(&self) -> Option<&[(Value, Value)]> {
        match self {
            Value::Dict(entries) => Some(entries),
            _ => None,
        }
    }

    /// The value under a string key of a dict; `None` when this is not a dict or has no such key
    pub fn get(&self, key: &str) -> Option<&Value> {
        let entries = self.as_dict()?;
        entries.iter().find(|(k, _)| matches!(k, Value::Str(s) if s == key)).map(|(_, v)| v)
    }

    /// The string, when this is a `str`
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The integer, when this is an i64-sized `int`
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// An opened checkpoint: its payload and the storages its tensors read
pub struct Pth {
    map: Mmap,
    archive: String,
    entries: HashMap<String, Range<usize>>,
    payload: Value,
}

impl Pth {
    /// Maps `path`, reads its zip directory and unpickles `data.pkl` in the restricted machine
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
        // SAFETY: the map is read-only and nothing writes through it; another process truncating the file while
        // it is mapped would fault this one, the hazard every mmap reader accepts
        let map = unsafe { Mmap::map(&file) }.wrap_err_with(|| format!("mapping {}", path.display()))?;
        Self::from_map(map).wrap_err_with(|| format!("reading checkpoint {}", path.display()))
    }

    fn from_map(map: Mmap) -> Result<Self> {
        let listed = read_zip(&map)?;
        // torch names the archive after the first entry's top directory (PyTorchStreamReader)
        let archive = listed
            .first()
            .and_then(|(name, _)| name.split_once('/'))
            .map(|(dir, _)| dir.to_string())
            .ok_or_else(|| eyre!("the zip holds no torch archive directory"))?;
        let mut entries = HashMap::with_capacity(listed.len());
        for (name, range) in listed {
            ensure!(!entries.contains_key(&name), "zip entry {name} appears twice");
            entries.insert(name, range);
        }

        let byteorder = entries
            .get(&format!("{archive}/byteorder"))
            .ok_or_else(|| eyre!("no {archive}/byteorder entry: refusing a checkpoint of unknown byte order"))?;
        ensure!(&map[byteorder.clone()] == b"little", "big-endian checkpoints are not supported");

        let pickle = entries.get(&format!("{archive}/data.pkl")).ok_or_else(|| eyre!("no {archive}/data.pkl"))?;
        ensure!(pickle.len() <= MAX_PICKLE_BYTES, "data.pkl is {} bytes, over the limit", pickle.len());
        let storage_len = |key: &str| entries.get(&format!("{archive}/data/{key}")).map(Range::len);
        let payload = unpickle(&map[pickle.clone()], &storage_len).wrap_err("unpickling data.pkl")?;
        Ok(Self { map, archive, entries, payload })
    }

    /// The unpickled payload
    pub fn payload(&self) -> &Value {
        &self.payload
    }

    /// The tensor's elements, copied out of the mapped storage in row-major order
    pub fn tensor_f32(&self, tensor: &TensorRef) -> Result<Vec<f32>> {
        let Dtype::F32 = tensor.dtype;
        let range = self
            .entries
            .get(&format!("{}/data/{}", self.archive, tensor.storage))
            .ok_or_else(|| eyre!("no storage {} in this checkpoint", tensor.storage))?;
        // a TensorRef carries no tie to the file it came from, so its bounds are checked against this one again
        let start = tensor.offset.checked_mul(4).ok_or_else(|| eyre!("tensor offset overflows"))?;
        let end = tensor.numel().checked_mul(4).and_then(|n| n.checked_add(start)).filter(|&end| end <= range.len());
        let end = end.ok_or_else(|| eyre!("tensor runs past storage {}", tensor.storage))?;
        let bytes = &self.map[range.start + start..range.start + end];
        // the length is a multiple of 4 by construction, so no remainder is dropped
        Ok(bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
    }
}

/// Little-endian reads over a byte slice that fail instead of running past its end
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&end| end <= self.buf.len());
        let end = end.ok_or_else(|| eyre!("truncated: {n} bytes wanted at byte {}", self.pos))?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// A length or offset from the file, as a usize
    fn size(&mut self, value: u64) -> Result<usize> {
        usize::try_from(value).map_err(|_| eyre!("size {value} at byte {} does not fit in memory", self.pos))
    }
}

const EOCD: u32 = 0x0605_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_EOCD: u32 = 0x0606_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;

/// The entries of a zip of stored entries, in central-directory order, as (name, data range)
///
/// Hand-written rather than the `zip` crate: torch writes every entry stored, so a central-directory walk with
/// zip64 support is all a checkpoint needs, and refusing compression, encryption and multi-disk archives
/// outright keeps the surface an untrusted file can reach small
fn read_zip(buf: &[u8]) -> Result<Vec<(String, Range<usize>)>> {
    let eocd = find_eocd(buf)?;
    let mut r = Reader::at(buf, eocd + 4);
    let (disk, cd_disk, disk_entries, entries) = (r.u16()?, r.u16()?, r.u16()?, r.u16()?);
    let (cd_size, cd_offset) = (r.u32()?, r.u32()?);
    let mut count = u64::from(entries);
    let mut cd = (u64::from(cd_offset), u64::from(cd_size));
    match zip64_eocd(buf, eocd)? {
        Some((n, offset, size)) => (count, cd) = (n, (offset, size)),
        None => {
            ensure!(disk == 0 && cd_disk == 0 && disk_entries == entries, "multi-disk zips are not supported");
            ensure!(
                entries != u16::MAX && cd_size != u32::MAX && cd_offset != u32::MAX,
                "zip64 marker without zip64 record"
            );
        }
    }

    let start = r.size(cd.0)?;
    let len = r.size(cd.1)?;
    let end =
        start.checked_add(len).filter(|&end| end <= eocd).ok_or_else(|| eyre!("central directory out of range"))?;
    let mut r = Reader::at(&buf[..end], start);
    let mut out = Vec::new();
    for _ in 0..count {
        out.push(central_entry(buf, &mut r)?);
    }

    ensure!(r.pos == end, "central directory holds more than its {count} entries");
    Ok(out)
}

fn find_eocd(buf: &[u8]) -> Result<usize> {
    ensure!(buf.len() >= 22, "not a zip: {} bytes", buf.len());
    // the record sits at the end, followed only by a comment of at most 64 KiB whose length it gives
    let lowest = buf.len().saturating_sub(22 + usize::from(u16::MAX));
    for pos in (lowest..=buf.len() - 22).rev() {
        let sig = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        let comment = usize::from(u16::from_le_bytes([buf[pos + 20], buf[pos + 21]]));
        if sig == EOCD && pos + 22 + comment == buf.len() {
            return Ok(pos);
        }
    }

    bail!("not a zip: no end-of-central-directory record")
}

/// The zip64 record's (entries, central directory offset, size) when the archive has one
fn zip64_eocd(buf: &[u8], eocd: usize) -> Result<Option<(u64, u64, u64)>> {
    let Some(locator) = eocd.checked_sub(20) else { return Ok(None) };
    let mut r = Reader::at(buf, locator);
    if r.u32()? != ZIP64_LOCATOR {
        return Ok(None);
    }

    let (locator_disk, offset, disks) = (r.u32()?, r.u64()?, r.u32()?);
    ensure!(locator_disk == 0 && disks == 1, "multi-disk zips are not supported");
    let mut r = Reader::at(&buf[..locator], r.size(offset)?);
    ensure!(r.u32()? == ZIP64_EOCD, "zip64 locator points at no zip64 record");
    let _record_size = r.u64()?;
    let _versions = r.u32()?;
    let (disk, cd_disk, disk_entries, entries) = (r.u32()?, r.u32()?, r.u64()?, r.u64()?);
    ensure!(disk == 0 && cd_disk == 0 && disk_entries == entries, "multi-disk zips are not supported");
    let (size, offset) = (r.u64()?, r.u64()?);
    Ok(Some((entries, offset, size)))
}

/// One central-directory entry and the data range its local header gives
fn central_entry(buf: &[u8], r: &mut Reader) -> Result<(String, Range<usize>)> {
    ensure!(r.u32()? == CENTRAL, "bad central directory entry at byte {}", r.pos - 4);
    let _versions = r.u32()?;
    let (flags, method) = (r.u16()?, r.u16()?);
    let _time_date_crc = r.take(8)?;
    let (mut csize, mut size) = (u64::from(r.u32()?), u64::from(r.u32()?));
    let (name_len, extra_len, comment_len) = (r.u16()?, r.u16()?, r.u16()?);
    let disk = r.u16()?;
    let _attributes = r.take(6)?;
    let mut local = u64::from(r.u32()?);
    let name = std::str::from_utf8(r.take(name_len.into())?).wrap_err("zip entry name is not UTF-8")?.to_string();
    let extra = r.take(extra_len.into())?;
    r.take(comment_len.into())?;

    // bit 0 and bit 6 mark encryption; CRCs are not checked, since anyone who can craft the file can fix them
    ensure!(flags & 0x41 == 0, "zip entry {name} is encrypted");
    ensure!(method == 0, "zip entry {name} is compressed (method {method}); torch writes stored entries only");
    ensure!(disk == 0 || disk == u16::MAX, "zip entry {name} is on another disk");

    // the zip64 extra field holds, in this order, each 32-bit field that is saturated
    let mut x = Reader::at(extra, 0);
    while x.pos < extra.len() {
        let (id, len) = (x.u16()?, x.u16()?);
        let body = x.take(len.into())?;
        if id != 1 {
            continue;
        }

        let mut z = Reader::at(body, 0);
        for field in [&mut size, &mut csize, &mut local] {
            if *field == u64::from(u32::MAX) {
                *field = z.u64()?;
            }
        }
    }

    ensure!(csize == size, "zip entry {name}: stored size {csize} differs from its size {size}");
    let mut l = Reader::at(buf, r.size(local)?);
    ensure!(l.u32()? == LOCAL, "zip entry {name}: no local header at {local}");
    let _header = l.take(22)?;
    let (local_name_len, local_extra_len) = (l.u16()?, l.u16()?);
    ensure!(l.take(local_name_len.into())? == name.as_bytes(), "zip entry {name}: local header names another file");
    l.take(local_extra_len.into())?;
    let start = l.pos;
    let len = l.size(size)?;
    let end = start.checked_add(len).filter(|&end| end <= buf.len());
    let end = end.ok_or_else(|| eyre!("zip entry {name} runs past the end of the file"))?;
    Ok((name, start..end))
}

type Id = usize;

/// What a class reference does when the machine calls it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    OrderedDict,
    RebuildTensorV2,
    RebuildParameter,
    Storage(Dtype),
    /// Anything else: calling it records an inert object
    Inert,
}

/// The fixed table of names the machine gives meaning to; every other name is inert
fn role(module: &str, name: &str) -> Role {
    match (module, name) {
        ("collections", "OrderedDict") => Role::OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2") => Role::RebuildTensorV2,
        ("torch._utils", "_rebuild_parameter") => Role::RebuildParameter,
        ("torch", "FloatStorage") => Role::Storage(Dtype::F32),
        _ => Role::Inert,
    }
}

/// A storage `persistent_load` resolved, before a tensor views it
#[derive(Debug, Clone)]
struct Storage {
    key: String,
    dtype: Dtype,
    numel: usize,
}

struct DictObj {
    entries: Vec<(Id, Id)>,
    keys: HashSet<Key>,
}

struct ObjectObj {
    callee: Id,
    args: Id,
    state: Option<Id>,
    items: Vec<Id>,
    entries: Vec<(Id, Id)>,
}

/// The machine's objects; containers refer to others by index so the memo can share and mutate them
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    BigInt(Vec<u8>),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Id>),
    Tuple(Vec<Id>),
    Dict(DictObj),
    Global { path: String, role: Role },
    Object(ObjectObj),
    Storage(Storage),
    Tensor(TensorRef),
}

/// A dict key, hashed as Python compares keys: `True == 1 == 1.0`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    None,
    Int(i64),
    BigInt(Vec<u8>),
    Float(u64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Key>),
    Global(String),
}

/// Runs `data` in the restricted machine; `storage_len` gives a storage key's entry length, if it exists
fn unpickle(data: &[u8], storage_len: &dyn Fn(&str) -> Option<usize>) -> Result<Value> {
    let mut machine = Machine {
        r: Reader::at(data, 0),
        objs: Vec::new(),
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        storage_len,
        storages: HashMap::new(),
    };
    let root = machine.run()?;
    let mut budget = MAX_VALUES;
    machine.value(root, 0, &mut budget)
}

struct Machine<'a> {
    r: Reader<'a>,
    objs: Vec<Obj>,
    stack: Vec<Id>,
    marks: Vec<usize>,
    memo: HashMap<u32, Id>,
    storage_len: &'a dyn Fn(&str) -> Option<usize>,
    storages: HashMap<String, (Dtype, usize)>,
}

impl Machine<'_> {
    fn run(&mut self) -> Result<Id> {
        loop {
            let at = self.r.pos;
            let op = self.r.u8()?;
            if op == b'.' {
                return self.pop();
            }

            self.step(op).wrap_err_with(|| format!("pickle opcode {op:#04x} at byte {at}"))?;
        }
    }

    fn step(&mut self, op: u8) -> Result<()> {
        match op {
            0x80 => {
                let proto = self.r.u8()?;
                ensure!((2..=5).contains(&proto), "pickle protocol {proto} is not supported");
            }
            0x95 => {
                // FRAME only groups opcodes for buffered readers
                self.r.u64()?;
            }
            b'(' => {
                ensure!(self.marks.len() < MAX_OBJECTS, "too many marks");
                self.marks.push(self.stack.len());
            }
            b'0' => {
                self.pop()?;
            }
            b'1' => {
                self.pop_mark()?;
            }
            b'2' => {
                let top = self.top()?;
                self.push_id(top)?;
            }
            b'N' => self.push(Obj::None)?,
            0x88 => self.push(Obj::Bool(true))?,
            0x89 => self.push(Obj::Bool(false))?,
            b'J' => {
                let v = i32::from_le_bytes(self.r.array()?);
                self.push(Obj::Int(v.into()))?;
            }
            b'K' => {
                let v = self.r.u8()?;
                self.push(Obj::Int(v.into()))?;
            }
            b'M' => {
                let v = self.r.u16()?;
                self.push(Obj::Int(v.into()))?;
            }
            0x8a => {
                let n = usize::from(self.r.u8()?);
                let bytes = self.r.take(n)?;
                self.push(long(bytes))?;
            }
            0x8b => {
                let n = self.r.u32()?;
                let n = self.r.size(n.into())?;
                let bytes = self.r.take(n)?;
                self.push(long(bytes))?;
            }
            b'G' => {
                // BINFLOAT is the one big-endian field in a pickle
                let v = f64::from_be_bytes(self.r.array()?);
                self.push(Obj::Float(v))?;
            }
            b'X' | 0x8c | 0x8d => {
                let n = self.len_prefix(op == 0x8c, op == 0x8d)?;
                let s = std::str::from_utf8(self.r.take(n)?).wrap_err("string is not UTF-8")?;
                self.push(Obj::Str(s.to_string()))?;
            }
            b'B' | b'C' | 0x8e => {
                let n = self.len_prefix(op == b'C', op == 0x8e)?;
                let bytes = self.r.take(n)?.to_vec();
                self.push(Obj::Bytes(bytes))?;
            }
            b')' => self.push(Obj::Tuple(Vec::new()))?,
            b't' => {
                let items = self.pop_mark()?;
                self.push(Obj::Tuple(items))?;
            }
            0x85..=0x87 => {
                let n = usize::from(op - 0x84);
                let at = self.stack.len().checked_sub(n).ok_or_else(|| eyre!("stack underflow"))?;
                let items = self.stack.split_off(at);
                self.push(Obj::Tuple(items))?;
            }
            b']' => self.push(Obj::List(Vec::new()))?,
            b'l' => {
                let items = self.pop_mark()?;
                self.push(Obj::List(items))?;
            }
            b'a' => {
                let v = self.pop()?;
                self.append(vec![v])?;
            }
            b'e' => {
                let items = self.pop_mark()?;
                self.append(items)?;
            }
            b'}' => self.push(Obj::Dict(DictObj { entries: Vec::new(), keys: HashSet::new() }))?,
            b'd' => {
                let items = self.pop_mark()?;
                self.push(Obj::Dict(DictObj { entries: Vec::new(), keys: HashSet::new() }))?;
                self.set_items(items)?;
            }
            b's' => {
                let v = self.pop()?;
                let k = self.pop()?;
                self.set_items(vec![k, v])?;
            }
            b'u' => {
                let items = self.pop_mark()?;
                self.set_items(items)?;
            }
            b'q' => {
                let i = self.r.u8()?;
                self.memo_put(i.into())?;
            }
            b'r' => {
                let i = self.r.u32()?;
                self.memo_put(i)?;
            }
            0x94 => {
                let i = u32::try_from(self.memo.len()).map_err(|_| eyre!("memo full"))?;
                self.memo_put(i)?;
            }
            b'h' => {
                let i = self.r.u8()?;
                self.memo_get(i.into())?;
            }
            b'j' => {
                let i = self.r.u32()?;
                self.memo_get(i)?;
            }
            b'c' => {
                let module = self.line()?;
                let name = self.line()?;
                self.global(&module, &name)?;
            }
            0x93 => {
                let name = self.pop_str()?;
                let module = self.pop_str()?;
                self.global(&module, &name)?;
            }
            b'R' => {
                let args = self.pop()?;
                let callee = self.pop()?;
                let obj = self.reduce(callee, args)?;
                self.push(obj)?;
            }
            0x81 => {
                let args = self.pop()?;
                let class = self.pop()?;
                let obj = self.new_obj(class, args)?;
                self.push(obj)?;
            }
            b'b' => {
                let state = self.pop()?;
                let target = self.top()?;
                let kind = self.kind(target);
                let Obj::Object(object) = &mut self.objs[target] else {
                    bail!("BUILD on a {kind}, only inert objects take state")
                };
                ensure!(object.state.is_none(), "BUILD twice on one object");
                object.state = Some(state);
            }
            b'Q' => {
                let pid = self.pop()?;
                let storage = self.persistent_load(pid)?;
                self.push(Obj::Storage(storage))?;
            }
            _ => bail!("opcode not supported by the restricted unpickler"),
        }

        Ok(())
    }

    fn len_prefix(&mut self, short: bool, long: bool) -> Result<usize> {
        let n = match (short, long) {
            (true, _) => u64::from(self.r.u8()?),
            (_, true) => self.r.u64()?,
            _ => u64::from(self.r.u32()?),
        };
        self.r.size(n)
    }

    fn line(&mut self) -> Result<String> {
        let rest = &self.r.buf[self.r.pos..];
        let n = rest.iter().take(256).position(|&b| b == b'\n').ok_or_else(|| eyre!("GLOBAL name too long"))?;
        let bytes = self.r.take(n + 1)?;
        let line = &bytes[..n];
        ensure!(line.iter().all(|b| b.is_ascii_graphic()), "GLOBAL name is not printable ASCII");
        Ok(line.iter().map(|&b| char::from(b)).collect())
    }

    fn push(&mut self, obj: Obj) -> Result<()> {
        ensure!(self.objs.len() < MAX_OBJECTS, "the pickle makes more than {MAX_OBJECTS} objects");
        self.objs.push(obj);
        self.push_id(self.objs.len() - 1)
    }

    fn push_id(&mut self, id: Id) -> Result<()> {
        ensure!(self.stack.len() < MAX_OBJECTS, "the pickle's stack grows past {MAX_OBJECTS}");
        self.stack.push(id);
        Ok(())
    }

    fn pop(&mut self) -> Result<Id> {
        let floor = self.marks.last().copied().unwrap_or(0);
        ensure!(self.stack.len() > floor, "stack underflow");
        self.stack.pop().ok_or_else(|| eyre!("stack underflow"))
    }

    fn top(&self) -> Result<Id> {
        let floor = self.marks.last().copied().unwrap_or(0);
        ensure!(self.stack.len() > floor, "stack underflow");
        self.stack.last().copied().ok_or_else(|| eyre!("stack underflow"))
    }

    fn pop_mark(&mut self) -> Result<Vec<Id>> {
        let mark = self.marks.pop().ok_or_else(|| eyre!("no mark"))?;
        Ok(self.stack.split_off(mark))
    }

    fn pop_str(&mut self) -> Result<String> {
        let id = self.pop()?;
        match &self.objs[id] {
            Obj::Str(s) => Ok(s.clone()),
            _ => bail!("STACK_GLOBAL needs strings, got a {}", self.kind(id)),
        }
    }

    fn memo_put(&mut self, i: u32) -> Result<()> {
        let top = self.top()?;
        ensure!(self.memo.len() < MAX_OBJECTS || self.memo.contains_key(&i), "memo grows past {MAX_OBJECTS}");
        self.memo.insert(i, top);
        Ok(())
    }

    fn memo_get(&mut self, i: u32) -> Result<()> {
        let id = *self.memo.get(&i).ok_or_else(|| eyre!("memo {i} is empty"))?;
        self.push_id(id)
    }

    fn global(&mut self, module: &str, name: &str) -> Result<()> {
        let path = format!("{module}.{name}");
        self.push(Obj::Global { role: role(module, name), path })
    }

    fn kind(&self, id: Id) -> &'static str {
        match &self.objs[id] {
            Obj::None => "None",
            Obj::Bool(_) => "bool",
            Obj::Int(_) | Obj::BigInt(_) => "int",
            Obj::Float(_) => "float",
            Obj::Str(_) => "str",
            Obj::Bytes(_) => "bytes",
            Obj::List(_) => "list",
            Obj::Tuple(_) => "tuple",
            Obj::Dict(_) => "dict",
            Obj::Global { .. } => "global",
            Obj::Object(_) => "inert object",
            Obj::Storage(_) => "storage",
            Obj::Tensor(_) => "tensor",
        }
    }

    fn append(&mut self, items: Vec<Id>) -> Result<()> {
        let target = self.top()?;
        let kind = self.kind(target);
        match &mut self.objs[target] {
            Obj::List(list) => list.extend(items),
            Obj::Object(object) => object.items.extend(items),
            _ => bail!("APPEND to a {kind}"),
        }

        Ok(())
    }

    fn set_items(&mut self, items: Vec<Id>) -> Result<()> {
        ensure!(items.len().is_multiple_of(2), "SETITEMS with an odd number of values");
        let target = self.top()?;
        let pairs: Vec<(Id, Id)> = items.as_chunks::<2>().0.iter().map(|&[k, v]| (k, v)).collect();
        if let Obj::Object(object) = &mut self.objs[target] {
            object.entries.extend(pairs);
            return Ok(());
        }

        ensure!(matches!(self.objs[target], Obj::Dict(_)), "SETITEM on a {}", self.kind(target));
        let keys = pairs.iter().map(|&(k, _)| self.key(k, 0)).collect::<Result<Vec<_>>>()?;
        let Obj::Dict(dict) = &mut self.objs[target] else { bail!("SETITEM target changed") };
        for (key, pair) in keys.into_iter().zip(pairs) {
            // Python would replace the value in place; no writer emits a key twice, so a repeat is refused
            ensure!(dict.keys.insert(key.clone()), "dict key {key:?} set twice");
            dict.entries.push(pair);
        }

        Ok(())
    }

    fn key(&self, id: Id, depth: usize) -> Result<Key> {
        ensure!(depth < MAX_DEPTH, "dict key nested too deep");
        Ok(match &self.objs[id] {
            Obj::None => Key::None,
            Obj::Bool(b) => Key::Int((*b).into()),
            Obj::Int(i) => Key::Int(*i),
            Obj::BigInt(b) => Key::BigInt(b.clone()),
            // an integral float equals the int, as in Python; the cast is exact inside the checked range
            Obj::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => Key::Int(*f as i64),
            Obj::Float(f) => Key::Float(f.to_bits()),
            Obj::Str(s) => Key::Str(s.clone()),
            Obj::Bytes(b) => Key::Bytes(b.clone()),
            Obj::Tuple(items) => Key::Tuple(items.iter().map(|&i| self.key(i, depth + 1)).collect::<Result<_>>()?),
            Obj::Global { path, .. } => Key::Global(path.clone()),
            _ => bail!("a {} as a dict key", self.kind(id)),
        })
    }

    fn reduce(&mut self, callee: Id, args: Id) -> Result<Obj> {
        let role = match &self.objs[callee] {
            Obj::Global { role, .. } => *role,
            Obj::Object(_) => Role::Inert,
            _ => bail!("REDUCE calls a {}", self.kind(callee)),
        };
        match role {
            Role::OrderedDict => {
                ensure!(matches!(&self.objs[args], Obj::Tuple(a) if a.is_empty()), "OrderedDict with arguments");
                Ok(Obj::Dict(DictObj { entries: Vec::new(), keys: HashSet::new() }))
            }
            Role::RebuildTensorV2 => self.rebuild_tensor(args).map(Obj::Tensor),
            Role::RebuildParameter => self.rebuild_parameter(args).map(Obj::Tensor),
            Role::Storage(_) | Role::Inert => Ok(self.inert(callee, args)),
        }
    }

    fn new_obj(&mut self, class: Id, args: Id) -> Result<Obj> {
        let Obj::Global { role, path } = &self.objs[class] else { bail!("NEWOBJ on a {}", self.kind(class)) };
        match role {
            // `OrderedDict.__new__` ignores its arguments; NEWOBJ does not call `__init__`
            Role::OrderedDict => Ok(Obj::Dict(DictObj { entries: Vec::new(), keys: HashSet::new() })),
            Role::Storage(_) | Role::Inert => Ok(self.inert(class, args)),
            Role::RebuildTensorV2 | Role::RebuildParameter => bail!("NEWOBJ on the function {path}"),
        }
    }

    fn inert(&self, callee: Id, args: Id) -> Obj {
        Obj::Object(ObjectObj { callee, args, state: None, items: Vec::new(), entries: Vec::new() })
    }

    fn tuple(&self, id: Id, what: &str) -> Result<&[Id]> {
        match &self.objs[id] {
            Obj::Tuple(items) => Ok(items),
            _ => bail!("{what} is a {}, expected a tuple", self.kind(id)),
        }
    }

    fn int(&self, id: Id, what: &str) -> Result<usize> {
        match &self.objs[id] {
            Obj::Int(i) => usize::try_from(*i).map_err(|_| eyre!("{what} is {i}, expected a non-negative int")),
            _ => bail!("{what} is a {}, expected an int", self.kind(id)),
        }
    }

    /// No hooks or tensor metadata: an empty dict or `None`
    fn empty_or_none(&self, id: Id, what: &str) -> Result<()> {
        match &self.objs[id] {
            Obj::None => Ok(()),
            Obj::Dict(d) if d.entries.is_empty() => Ok(()),
            _ => bail!("{what} is set; only plain tensors are supported"),
        }
    }

    /// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks[, metadata])`
    fn rebuild_tensor(&self, args: Id) -> Result<TensorRef> {
        let args = self.tuple(args, "_rebuild_tensor_v2 arguments")?;
        let [storage, offset, size, stride, requires_grad, hooks, rest @ ..] = args else {
            bail!("_rebuild_tensor_v2 with {} arguments", args.len())
        };
        ensure!(rest.len() <= 1, "_rebuild_tensor_v2 with {} arguments", args.len());
        if let Some(&metadata) = rest.first() {
            self.empty_or_none(metadata, "tensor metadata")?;
        }

        let Obj::Storage(storage) = &self.objs[*storage] else {
            bail!("_rebuild_tensor_v2 on a {}, expected a storage", self.kind(*storage))
        };
        ensure!(matches!(self.objs[*requires_grad], Obj::Bool(_)), "requires_grad is not a bool");
        self.empty_or_none(*hooks, "backward hooks")?;
        let offset = self.int(*offset, "storage offset")?;
        let shape = self.tuple(*size, "size")?;
        let stride = self.tuple(*stride, "stride")?;
        ensure!(shape.len() <= MAX_NDIM, "{} dimensions", shape.len());
        ensure!(shape.len() == stride.len(), "size has {} dims, stride {}", shape.len(), stride.len());
        let shape = shape.iter().map(|&d| self.int(d, "size")).collect::<Result<Vec<_>>>()?;
        let stride = stride.iter().map(|&d| self.int(d, "stride")).collect::<Result<Vec<_>>>()?;

        let numel = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| eyre!("size overflows"))?;
        ensure!(contiguous(&shape, &stride), "tensor of size {shape:?} has strides {stride:?}, not contiguous");
        let end = offset.checked_add(numel).ok_or_else(|| eyre!("offset overflows"))?;
        ensure!(
            numel == 0 || end <= storage.numel,
            "tensor of size {shape:?} at offset {offset} runs past storage {} of {} elements",
            storage.key,
            storage.numel
        );
        Ok(TensorRef { storage: storage.key.clone(), dtype: storage.dtype, offset, shape })
    }

    /// `_rebuild_parameter(data, requires_grad, backward_hooks)`: the parameter is its tensor
    fn rebuild_parameter(&self, args: Id) -> Result<TensorRef> {
        let args = self.tuple(args, "_rebuild_parameter arguments")?;
        let [data, requires_grad, hooks] = args else { bail!("_rebuild_parameter with {} arguments", args.len()) };
        let Obj::Tensor(tensor) = &self.objs[*data] else { bail!("_rebuild_parameter on a {}", self.kind(*data)) };
        ensure!(matches!(self.objs[*requires_grad], Obj::Bool(_)), "requires_grad is not a bool");
        self.empty_or_none(*hooks, "backward hooks")?;
        Ok(tensor.clone())
    }

    /// `('storage', storage_type, key, location, numel)`, checked against the storage's zip entry
    fn persistent_load(&mut self, pid: Id) -> Result<Storage> {
        let pid = self.tuple(pid, "persistent id")?;
        let &[tag, class, key, location, numel] = pid else { bail!("persistent id of {} fields", pid.len()) };
        ensure!(matches!(&self.objs[tag], Obj::Str(s) if s == "storage"), "persistent id is not a storage");
        let dtype = match &self.objs[class] {
            Obj::Global { role: Role::Storage(dtype), .. } => *dtype,
            Obj::Global { path, .. } => bail!("storage type {path} is not supported (float32 only)"),
            _ => bail!("storage type is a {}", self.kind(class)),
        };
        let key = match &self.objs[key] {
            Obj::Str(s) => s.clone(),
            _ => bail!("storage key is a {}", self.kind(key)),
        };
        ensure!(
            !key.is_empty() && key.len() <= 64 && key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "storage key {key:?} is not a plain name"
        );
        ensure!(matches!(self.objs[location], Obj::Str(_)), "storage location is not a string");
        let numel = self.int(numel, "storage size")?;

        let bytes = numel.checked_mul(dtype.size()).ok_or_else(|| eyre!("storage {key} size overflows"))?;
        let entry = (self.storage_len)(&key).ok_or_else(|| eyre!("storage {key} has no zip entry"))?;
        ensure!(entry == bytes, "storage {key} of {numel} elements needs {bytes} bytes, its entry holds {entry}");
        let seen = self.storages.entry(key.clone()).or_insert((dtype, numel));
        ensure!(*seen == (dtype, numel), "storage {key} is loaded with two types or sizes");
        Ok(Storage { key, dtype, numel })
    }

    /// The owned value of object `id`, bounded in depth and total size
    fn value(&self, id: Id, depth: usize, budget: &mut usize) -> Result<Value> {
        ensure!(depth < MAX_DEPTH, "payload nested deeper than {MAX_DEPTH} (or cyclic)");
        *budget = budget.checked_sub(1).ok_or_else(|| eyre!("payload expands past {MAX_VALUES} values"))?;
        let mut all = |ids: &[Id]| ids.iter().map(|&i| self.value(i, depth + 1, budget)).collect::<Result<Vec<_>>>();
        Ok(match &self.objs[id] {
            Obj::None => Value::None,
            Obj::Bool(b) => Value::Bool(*b),
            Obj::Int(i) => Value::Int(*i),
            Obj::BigInt(b) => Value::BigInt(b.clone()),
            Obj::Float(f) => Value::Float(*f),
            Obj::Str(s) => Value::Str(s.clone()),
            Obj::Bytes(b) => Value::Bytes(b.clone()),
            Obj::List(items) => Value::List(all(items)?),
            Obj::Tuple(items) => Value::Tuple(all(items)?),
            Obj::Dict(dict) => Value::Dict(self.pairs(&dict.entries, depth, budget)?),
            Obj::Global { path, .. } => Value::Global(path.clone()),
            Obj::Storage(s) => Value::Object(Box::new(Object {
                callee: Value::Global("torch.storage".into()),
                args: Value::Str(s.key.clone()),
                state: None,
                items: Vec::new(),
                entries: Vec::new(),
            })),
            Obj::Tensor(t) => Value::Tensor(t.clone()),
            Obj::Object(o) => {
                let callee = self.value(o.callee, depth + 1, budget)?;
                let args = self.value(o.args, depth + 1, budget)?;
                let state = o.state.map(|s| self.value(s, depth + 1, budget)).transpose()?;
                let items = o.items.iter().map(|&i| self.value(i, depth + 1, budget)).collect::<Result<_>>()?;
                let entries = self.pairs(&o.entries, depth, budget)?;
                Value::Object(Box::new(Object { callee, args, state, items, entries }))
            }
        })
    }

    fn pairs(&self, pairs: &[(Id, Id)], depth: usize, budget: &mut usize) -> Result<Vec<(Value, Value)>> {
        let pair = |&(k, v): &(Id, Id), budget: &mut usize| -> Result<(Value, Value)> {
            Ok((self.value(k, depth + 1, budget)?, self.value(v, depth + 1, budget)?))
        };
        pairs.iter().map(|p| pair(p, budget)).collect()
    }
}

/// LONG1/LONG4: little-endian two's complement of any length
fn long(bytes: &[u8]) -> Obj {
    if bytes.is_empty() {
        return Obj::Int(0);
    }

    if bytes.len() > 8 {
        return Obj::BigInt(bytes.to_vec());
    }

    let fill = if bytes[bytes.len() - 1] & 0x80 != 0 { 0xff } else { 0 };
    let mut buf = [fill; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    Obj::Int(i64::from_le_bytes(buf))
}

/// torch's `is_contiguous`: row-major strides, ignoring dimensions of size 1, and any strides for no elements
fn contiguous(shape: &[usize], stride: &[usize]) -> bool {
    if shape.contains(&0) {
        return true;
    }

    let mut expected = 1usize;
    for (&size, &step) in shape.iter().zip(stride).rev() {
        if size == 1 {
            continue;
        }

        if step != expected {
            return false;
        }

        // the shape's product was checked for overflow, so this partial product fits
        expected *= size;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pickle holding one tensor of `size` and `stride` over float storage "0" of `numel` elements
    fn tensor_pickle(numel: u8, size: &[u8], stride: &[u8]) -> Vec<u8> {
        let mut p =
            b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storagectorch\nFloatStorage\n".to_vec();
        p.extend(b"X\x01\x00\x00\x000X\x03\x00\x00\x00cpuK");
        p.extend([numel, b't', b'Q', b'K', 0, b'(']);
        for &d in size {
            p.extend([b'K', d]);
        }

        p.extend(b"t(");
        for &d in stride {
            p.extend([b'K', d]);
        }

        p.extend(b"t\x89ccollections\nOrderedDict\n)RtR.");
        p
    }

    fn storage_of(bytes: usize) -> impl Fn(&str) -> Option<usize> {
        move |key: &str| (key == "0").then_some(bytes)
    }

    #[test]
    fn builds_a_contiguous_tensor() -> Result<()> {
        let value = unpickle(&tensor_pickle(6, &[2, 3], &[3, 1]), &storage_of(24))?;
        let Value::Tensor(t) = value else { bail!("not a tensor: {value:?}") };
        assert_eq!((t.storage(), t.offset(), t.shape()), ("0", 0, &[2, 3][..]));
        Ok(())
    }

    #[test]
    fn refuses_bad_tensors() {
        let transposed = unpickle(&tensor_pickle(6, &[2, 3], &[1, 2]), &storage_of(24));
        assert!(format!("{:?}", transposed.err()).contains("not contiguous"));
        let too_big = unpickle(&tensor_pickle(4, &[2, 3], &[3, 1]), &storage_of(16));
        assert!(format!("{:?}", too_big.err()).contains("runs past storage"));
        let short_entry = unpickle(&tensor_pickle(6, &[2, 3], &[3, 1]), &storage_of(20));
        assert!(format!("{:?}", short_entry.err()).contains("its entry holds 20"));
    }

    #[test]
    fn unknown_globals_stay_inert() -> Result<()> {
        // os.system("ls") followed by BUILD: recorded, never called
        let value = unpickle(b"\x80\x02cos\nsystem\nX\x02\x00\x00\x00ls\x85R}b.", &storage_of(0))?;
        let Value::Object(object) = value else { bail!("not an object: {value:?}") };
        assert_eq!(object.callee, Value::Global("os.system".into()));
        assert_eq!(object.args, Value::Tuple(vec![Value::Str("ls".into())]));
        assert_eq!(object.state, Some(Value::Dict(Vec::new())));
        Ok(())
    }

    #[test]
    fn refuses_text_opcodes_and_blowups() {
        // INST names a class and calls it in one opcode
        let inst = unpickle(b"\x80\x02(ios\nsystem\n.", &storage_of(0));
        assert!(format!("{:?}", inst.err()).contains("not supported"));

        // each level is a pair of the level below: 2^40 leaves if materialised naively
        let mut bomb = b"\x80\x02]q\x00".to_vec();
        for level in 0..40u8 {
            bomb.extend([b'h', level, b'h', level, 0x86, b'q', level + 1]);
        }

        bomb.push(b'.');
        assert!(format!("{:?}", unpickle(&bomb, &storage_of(0)).err()).contains("expands past"));
    }
}
