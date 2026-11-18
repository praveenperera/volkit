"""Compares each kernel entry of volkit's PTX with the same entry of another PTX file, such as inkrs's

Prints, per entry of the first file, whether its `.entry` text (signature and body) is identical to the second
file's up to what only names the entry's place in its module: block labels carry the entry's index in the file
(`$L__BB<index>_<block>`), shared arrays a hash of the defining crate (`__shared_mem_<hash>`), whose
declarations are compared by type, alignment and size instead, and comments name the crate. Exits non-zero
when any entry differs or is missing

    python3 -I scripts/compare-ptx-entries.py volkit/ptx/volkit.sm80.ptx /path/to/inkrs.sm80.ptx
"""

import re
import sys


def shared_decls(text: str) -> dict[str, str]:
    """Each module-scope shared array's declaration without its name"""
    decls = {}
    for m in re.finditer(r"^(?:\.visible )?\.shared (.*?)(__shared_mem_\w+)(\[\d+\])?;", text, re.M):
        decls[m.group(2)] = m.group(1) + (m.group(3) or "")
    return decls


def normalise(body: str, decls: dict[str, str]) -> str:
    body = re.sub(r"\s*//.*", "", body)
    body = re.sub(r"\$L__BB\d+_", "$L__BB_", body)
    names = []
    for name in re.findall(r"__shared_mem_\w+", body):
        if name not in names:
            names.append(name)
    for i, name in enumerate(names):
        body = body.replace(name, f"__shared{i}<{decls.get(name, '?')}>")
    return body


def entries(path: str) -> dict[str, str]:
    text = open(path).read()
    decls = shared_decls(text)
    out = {}
    for m in re.finditer(r"^\.visible \.entry (\w+)\(", text, re.M):
        start = m.start()
        # the body ends at the first line holding only "}" after the opening brace
        end = text.index("\n}\n", text.index("{", m.end())) + 3
        out[m.group(1)] = normalise(text[start:end], decls)
    return out


def main() -> int:
    ours, theirs = entries(sys.argv[1]), entries(sys.argv[2])
    bad = 0
    for name, body in ours.items():
        other = theirs.get(name)
        if other is None:
            print(f"MISSING {name}")
            bad += 1
            continue
        same = body == other
        print(f"{'same' if same else 'DIFF'} {name} ({body.count(chr(10))} lines)")
        bad += not same
    print(f"{len(ours)} entries, {bad} differ or missing; {len(theirs) - len(ours)} entries only in the second file")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
