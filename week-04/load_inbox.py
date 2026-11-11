"""Place the Hartwell sample emails into a Gmail sandbox inbox as unread messages.

Nothing is sent. Each .eml file in hartwell-hr/sample-emails is added straight into the mailbox
with IMAP APPEND, unread, dated today. Messages already loaded (matched by Message-ID) are skipped,
so running it twice doesn't create duplicates; use --force to add them again.

The login comes from Agent-Email.txt in the folder above the repository, as two labelled lines:
    EMAIL_ADDRESS=your.sandbox.address@gmail.com
    APP_PASSWORD=abcdefghijklmnop
("email: ..." and "password: ..." work too, as does the address on one line and the password on the next.)
For Gmail the password must be an app password, and IMAP must be turned on in Gmail's settings.
The password is never printed or saved.

Usage (from the top of the repository):
    .venv/bin/python week-04/load_inbox.py             # load the sample emails
    .venv/bin/python week-04/load_inbox.py --dry-run   # show what would be loaded, without signing in
"""

import argparse
import imaplib
import re
import sys
import time
from email import message_from_bytes, policy
from email.utils import formatdate
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMAILS_DIR = HERE / "hartwell-hr" / "sample-emails"
LOGIN_FILE = HERE.parent.parent / "Agent-Email.txt"
IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"

ADDRESS_LABELS = {"email", "e mail", "address", "email address", "user", "username", "login", "account"}
PASSWORD_LABELS = {"password", "app password", "pass", "passcode"}


def read_login(path: Path) -> tuple[str, str]:
    """Return (address, password). Error messages never include the file's contents."""
    if not path.is_file():
        sys.exit(f"Couldn't find {path.name} in {path.parent}.")
    lines = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]

    address = password = None
    for line in lines:
        label, sep, value = line.partition(":") if ":" in line else line.partition("=")
        label = re.sub(r"[_-]+", " ", label.strip().lower())  # EMAIL_ADDRESS -> "email address"
        if sep and label in ADDRESS_LABELS:
            address = value.strip()
        elif sep and label in PASSWORD_LABELS:
            password = value.strip()
    if address is None and password is None and len(lines) >= 2:
        # Unlabelled: the address is the first line with an @, the password is the next line.
        at = next((i for i, l in enumerate(lines) if "@" in l), None)
        if at is not None and at + 1 < len(lines):
            address, password = lines[at], lines[at + 1]

    if not address or "@" not in address or not password:
        sys.exit(f"Couldn't read an address and password from {path.name}. Put the Gmail address on the "
                 "first line and the app password on the second (or label them 'email:' and 'password:').")
    # Google shows app passwords as four groups of four letters; the spaces aren't part of it.
    if re.fullmatch(r"[a-z]{4}( [a-z]{4}){3}", password):
        password = password.replace(" ", "")
    return address, password


def message_id(path: Path) -> str:
    return f"<hartwell-sample-{path.stem}@load-inbox.gbus562>"


def prepare(path: Path) -> bytes:
    """Return the email with today's date and a stable Message-ID, as CRLF bytes for IMAP."""
    msg = message_from_bytes(path.read_bytes(), policy=policy.default)
    for header in ("Date", "Message-ID"):
        del msg[header]
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = message_id(path)
    return msg.as_bytes(policy=policy.SMTP)


def already_loaded(imap: imaplib.IMAP4_SSL, path: Path) -> bool:
    status, data = imap.search(None, "HEADER", "Message-ID", f'"{message_id(path)}"')
    return status == "OK" and bool(data and data[0].split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Hartwell sample emails into a Gmail inbox as unread messages.")
    parser.add_argument("--dry-run", action="store_true", help="list the emails that would be loaded, without signing in")
    parser.add_argument("--force", action="store_true", help="add the emails even if they were loaded before")
    args = parser.parse_args()

    files = sorted(EMAILS_DIR.glob("*.eml"))
    if not files:
        sys.exit(f"No .eml files found in {EMAILS_DIR}.")

    if args.dry_run:
        for path in files:
            msg = message_from_bytes(prepare(path), policy=policy.default)
            print(f"  {path.name}: From {msg['From']} | Subject {msg['Subject']} | Date {msg['Date']}")
        print(f"Dry run: {len(files)} emails would be placed in {MAILBOX} as unread. Nothing was loaded.")
        return

    address, password = read_login(LOGIN_FILE)
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST)
    except OSError as e:
        sys.exit(f"Couldn't connect to {IMAP_HOST}: {e.__class__.__name__}. Check your internet connection.")
    try:
        try:
            imap.login(address, password)
        except imaplib.IMAP4.error:
            sys.exit(f"Gmail rejected the sandbox login. Check that {LOGIN_FILE.name} has a Gmail app "
                     "password (not the normal account password) and that IMAP is turned on in Gmail's settings.")
        finally:
            del password
        status, _ = imap.select(MAILBOX)
        if status != "OK":
            sys.exit(f"Couldn't open the {MAILBOX} mailbox.")

        added = skipped = 0
        for path in files:
            if not args.force and already_loaded(imap, path):
                print(f"  {path.name}: already in the inbox, skipped")
                skipped += 1
                continue
            # No flags means the message arrives unread; the internal date is now.
            status, _ = imap.append(MAILBOX, None, imaplib.Time2Internaldate(time.time()), prepare(path))
            if status != "OK":
                sys.exit(f"Gmail refused to add {path.name}.")
            print(f"  {path.name}: added as unread")
            added += 1
        print(f"Done: {added} added, {skipped} skipped, in the sandbox {MAILBOX}. Nothing was sent.")
    finally:
        try:
            imap.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


if __name__ == "__main__":
    main()
