"""Seal the instructor files into one encrypted archive that is safe to push.

    uv run scripts/instructor.py seal     # weeks/*/instructor/  ->  instructor.tar.gz.gpg
    uv run scripts/instructor.py open     # instructor.tar.gz.gpg  ->  weeks/*/instructor/

The repository is public (Colab and `pip install git+https://...` read it
with no login), so answer files cannot be pushed as they are. `weeks/*/instructor/`
stays gitignored; what is committed is `instructor.tar.gz.gpg`, a gzip tar of
every instructor folder encrypted with a passphrase (gpg --symmetric,
AES-256). Students can download it and cannot read it -- not even the file
names, since they are inside the tar.

gpg ships with Git for Windows (C:\\Program Files\\Git\\usr\\bin\\gpg.exe); it is
found there if it is not on PATH. No Python package is added, so uv.lock and
the students' environment are untouched.

The passphrase is asked for at the prompt (twice when sealing), or read from
SOC4180_INSTRUCTOR_PASS. It is never stored in the repository. Anyone can try
passphrases against the archive offline for as long as they like, so use a
long one -- four or five random words -- and keep it somewhere other than
this machine: without it the archive cannot be opened by anyone, including you.

Re-seal after changing an answer file, then commit the archive. `open` will not
overwrite a local file that differs from the archive's copy unless --force is
given, so a newer answer on this machine is never silently replaced.
"""

import argparse
import getpass
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "instructor.tar.gz.gpg"
GIT_GPG = Path(r"C:\Program Files\Git\usr\bin\gpg.exe")


def find_gpg() -> str:
    exe = shutil.which("gpg") or (str(GIT_GPG) if GIT_GPG.exists() else None)
    if exe is None:
        sys.exit("gpg not found: install Git for Windows, or gnupg on Linux/macOS")
    return exe


def passphrase(confirm: bool) -> str:
    env = os.environ.get("SOC4180_INSTRUCTOR_PASS")
    if env:
        return env
    p = getpass.getpass("instructor passphrase: ")
    if confirm and getpass.getpass("again: ") != p:
        sys.exit("the two passphrases differ; nothing was written")
    if not p:
        sys.exit("empty passphrase; nothing was written")
    return p


def gpg(args, secret: str, data: bytes = b"") -> bytes:
    """Run gpg with the passphrase on stdin (fd 0), never on the command line."""
    cmd = [find_gpg(), "--batch", "--yes", "--quiet", "--pinentry-mode", "loopback", "--passphrase-fd", "0", *args]
    r = subprocess.run(cmd, input=secret.encode() + b"\n" + data, capture_output=True)
    if r.returncode != 0:
        sys.exit("gpg failed: " + r.stderr.decode(errors="replace").strip())
    return r.stdout


def instructor_files():
    return sorted(p for d in ROOT.glob("weeks/*/instructor") for p in d.rglob("*") if p.is_file())


def seal(archive: Path) -> int:
    files = instructor_files()
    if not files:
        sys.exit("no files under weeks/*/instructor/ -- nothing to seal")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    secret = passphrase(confirm=True)
    # gpg reads the passphrase from stdin, so the tar goes through a temp file.
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "instructor.tar.gz"
        plain.write_bytes(buf.getvalue())
        gpg(["--symmetric", "--cipher-algo", "AES256", "--output", str(archive), str(plain)], secret)
    for f in files:
        print(f"  sealed {f.relative_to(ROOT).as_posix()}")
    print(f"{len(files)} file(s) -> {archive.name} ({archive.stat().st_size} bytes). Commit it.")
    return 0


def open_(archive: Path, force: bool) -> int:
    if not archive.exists():
        sys.exit(f"{archive.name} not found")
    data = gpg(["--decrypt", str(archive)], passphrase(confirm=False))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or len(parts) < 3 or parts[2] != "instructor":
                sys.exit(f"refusing unexpected path in archive: {m.name}")
        clashes = [m.name for m in members
                   if (ROOT / m.name).exists() and (ROOT / m.name).read_bytes() != tar.extractfile(m).read()]
        if clashes and not force:
            for c in clashes:
                print(f"  differs locally: {c}")
            sys.exit("local files differ from the archive; seal them first, or re-run with --force to overwrite")
        for m in members:
            dest = ROOT / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())
            print(f"  opened {m.name}")
    print(f"{len(members)} file(s) restored.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["seal", "open"])
    ap.add_argument("--archive", type=Path, default=ARCHIVE, help="default: instructor.tar.gz.gpg at the repo root")
    ap.add_argument("--force", action="store_true", help="open: overwrite local files that differ")
    args = ap.parse_args(argv)
    return seal(args.archive) if args.action == "seal" else open_(args.archive, args.force)


if __name__ == "__main__":
    sys.exit(main())
