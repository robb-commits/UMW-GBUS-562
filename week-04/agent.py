"""A command-line agent that prepares replies to Hartwell Foods HR emails, one email at a time.

For each email, the agent starts hr_mcp_server.py in employee mode for the sender (so the only
leave balance it can see is the sender's own), offers Claude that server's tools plus save_reply,
and lets Claude work until it saves a reply. Nothing is ever sent.

  --email PATH   handle one .eml file; save_reply writes the draft to week-04/drafts/
  --inbox        sign in to the Gmail sandbox with IMAP and handle every unread email in the inbox,
                 each as its own agent run; save_reply creates a Gmail draft in the same thread.
                 A draft that needs HR review gets the "Needs HR Review" label, and its original
                 email gets the label and a star, so reviewers can spot it in the inbox list
  --save PATH    also write everything printed to this text file (UTF-8)

Usage (from the top of the repository):
    .venv/bin/python week-04/agent.py --email week-04/hartwell-hr/sample-emails/01-dana-vacation.eml
    .venv/bin/python week-04/agent.py --inbox --save week-04/runs/inbox-run.txt

The system prompt is read from hartwell-hr/system_prompt.txt on every run. The API key is read
from API-Key.txt in the folder above the repository, and the Gmail login from Agent-Email.txt there;
neither is printed or written to any file.
"""

import argparse
import asyncio
import imaplib
import json
import re
import sys
import time
from contextlib import redirect_stdout
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path

import anthropic
from mcp import Client, StdioServerParameters

import load_inbox

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
PREVIEW_CHARS = 300
REVIEW_TAG = "[NEEDS HR REVIEW]"
REVIEW_LABEL = "Needs HR Review"  # Gmail label added to flagged emails and drafts in --inbox mode

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
SERVER = HERE / "hr_mcp_server.py"
DRAFTS_DIR = HERE / "drafts"
KEY_FILES = [HERE.parent.parent / "API-Key.txt", HERE.parent.parent / "API Key.txt"]

SAVE_REPLY_TOOL = {
    "name": "save_reply",
    "description": ("Save your reply to the employee as a draft for the HR team to review. It is never "
                    "sent. Call it exactly once, when the reply is final."),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply_text": {"type": "string", "description": "The full reply to the employee, ready for HR to review."},
            "needs_human_review": {"type": "boolean",
                                   "description": "True if a person from HR must handle or check this email."},
        },
        "required": ["reply_text", "needs_human_review"],
    },
}


class Fail(Exception):
    """An error to report to the user and stop on."""


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------

class Email:
    def __init__(self, raw: bytes, label: str):
        msg = message_from_bytes(raw, policy=policy.default)
        self.label = label
        self.from_header = str(msg["From"] or "")
        self.name, self.address = parseaddr(self.from_header)
        self.address = self.address.strip().lower()
        self.subject = str(msg["Subject"] or "(no subject)")
        self.message_id = str(msg["Message-ID"] or "").strip()
        self.references = str(msg["References"] or "").strip()
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
        if body is not None and body.get_content_type() == "text/html":
            text = re.sub(r"<[^>]+>", " ", text)
        self.body = text.strip()

    def reply_subject(self, needs_review: bool) -> str:
        subject = self.subject if re.match(r"(?i)re:", self.subject) else f"Re: {self.subject}"
        return f"{REVIEW_TAG} {subject}" if needs_review else subject

    def prompt(self) -> str:
        return ("Prepare a reply to this employee email.\n\n"
                f"<email>\nFrom: {self.from_header}\nSubject: {self.subject}\n\n{self.body}\n</email>")


# ---------------------------------------------------------------------------
# Where save_reply puts the draft
# ---------------------------------------------------------------------------

class FileDrafts:
    """--email mode: write the draft to week-04/drafts/."""

    def save(self, email: Email, reply: str, needs_review: bool) -> str:
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        path = DRAFTS_DIR / f"{Path(email.label).stem}-reply.txt"
        lines = [f"To: {email.from_header}", f"Subject: {email.reply_subject(needs_review)}",
                 f"Needs human review: {'yes' if needs_review else 'no'}"]
        if email.message_id:
            lines.append(f"In-Reply-To: {email.message_id}")
        path.write_text("\n".join(lines) + "\n\n" + reply.strip() + "\n", encoding="utf-8")
        shown = path.relative_to(HERE.parent) if path.is_relative_to(HERE.parent) else path
        return f"Draft saved to {shown}. It was not sent."


class GmailDrafts:
    """--inbox mode: create a Gmail draft replying in the email's thread. Drafts only; nothing is sent."""

    def __init__(self, imap: imaplib.IMAP4_SSL, address: str):
        self.imap = imap
        self.address = address
        self.folder = drafts_folder(imap)
        # The inbox email being handled; run_inbox sets these before each agent run.
        self.original_uid: str | None = None
        self.original_thread: str | None = None

    def save(self, email: Email, reply: str, needs_review: bool) -> str:
        msg = EmailMessage(policy=policy.SMTP)
        msg["From"] = self.address
        msg["To"] = email.from_header
        msg["Subject"] = email.reply_subject(needs_review)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="hartwell-agent.gbus562")
        if email.message_id:
            msg["In-Reply-To"] = email.message_id
            msg["References"] = f"{email.references} {email.message_id}".strip()
        msg.set_content(reply.strip() + "\n")

        status, data = self.imap.append(self.folder, r"(\Draft)", imaplib.Time2Internaldate(time.time()), msg.as_bytes())
        if status != "OK":
            raise Fail("Gmail refused to create the draft.")
        result = f"Gmail draft created in {self.folder}, replying to {email.address}. It was not sent."
        match = re.search(rb"APPENDUID \d+ (\d+)", b" ".join(d for d in data if isinstance(d, bytes)))
        draft_uid = match.group(1).decode() if match else None
        if needs_review:
            result += " " + self.flag_for_review(draft_uid)
        same = self.same_thread(draft_uid)
        if same is not None:
            result += " It is in the same thread as the original email." if same else \
                      " Gmail did not put it in the same thread as the original email."
        return result

    def flag_for_review(self, draft_uid: str | None) -> str:
        """Label and star the original email and label the draft, so HR can find it in the inbox list.
        A failure here is reported but doesn't undo the draft."""
        label = f'"{REVIEW_LABEL}"'
        done, failed = [], []
        try:
            self.imap.create(label)  # Gmail answers NO if the label already exists, which is fine
            if self.original_uid:
                ok_label = self.imap.uid("STORE", self.original_uid, "+X-GM-LABELS", f"({label})")[0] == "OK"
                ok_star = self.imap.uid("STORE", self.original_uid, "+FLAGS", r"(\Flagged)")[0] == "OK"
                if ok_label and ok_star:
                    done.append("labelled and starred the original email")
                else:
                    failed.append("label and star the original email")
            if draft_uid:
                self.imap.select(quote(self.folder))
                ok = self.imap.uid("STORE", draft_uid, "+X-GM-LABELS", f"({label})")[0] == "OK"
                done.append("labelled the draft") if ok else failed.append("label the draft")
        except imaplib.IMAP4.error:
            failed.append("add the label or star (Gmail rejected the change)")  # e.g. labels hidden from IMAP
        finally:
            self.imap.select("INBOX")
        text = f"For HR review: {', '.join(done)} with '{REVIEW_LABEL}'." if done else ""
        if failed:
            text += f" Couldn't {' or '.join(failed)}; please flag it for HR review by hand."
        return text.strip()

    def same_thread(self, draft_uid: str | None) -> bool | None:
        """Compare the draft's Gmail thread id with the original's. None if Gmail didn't say."""
        if not draft_uid or not self.original_thread:
            return None
        try:
            self.imap.select(quote(self.folder), readonly=True)
            _, data = self.imap.uid("FETCH", draft_uid, "(X-GM-THRID)")
            found = re.search(rb"X-GM-THRID (\d+)", data[0] if data and isinstance(data[0], bytes) else b"")
            return bool(found) and found.group(1).decode() == self.original_thread
        finally:
            self.imap.select("INBOX")


def quote(folder: str) -> str:
    return f'"{folder}"' if " " in folder and not folder.startswith('"') else folder


def drafts_folder(imap: imaplib.IMAP4_SSL) -> str:
    """Find the Drafts folder by its \\Drafts flag, so it works whatever the account's language."""
    status, folders = imap.list()
    for line in folders or []:
        text = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
        if r"\Drafts" in text:
            name = re.search(r'"[^"]*" (.+)$', text)
            if name:
                return name.group(1).strip()
    return '"[Gmail]/Drafts"'


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def load_client() -> anthropic.Anthropic:
    for path in KEY_FILES:
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return anthropic.Anthropic(api_key=key)
    raise Fail(f"Couldn't find your API key. Put it in {KEY_FILES[0]} (the folder above the repository).")


def preview(text: str) -> str:
    text = text.strip()
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


async def run_agent(client: anthropic.Anthropic, email: Email, drafts) -> tuple[int, int]:
    """Handle one email. Returns (input tokens, output tokens)."""
    print(f"\n{'=' * 70}\nEmail: {email.label}\nFrom: {email.from_header}\nSubject: {email.subject}\n{'=' * 70}")
    if "@" not in email.address:
        print("Skipped: couldn't read a sender address from this email.")
        return 0, 0

    system = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")  # read fresh on every run
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER), "--as-employee", email.address])
    tokens_in = tokens_out = tool_calls = 0
    saved: dict | None = None

    async with Client(server) as mcp:
        mcp_tools = (await mcp.list_tools()).tools
        tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                 for t in mcp_tools] + [SAVE_REPLY_TOOL]
        print(f"Tools offered: {', '.join(t['name'] for t in tools)}")
        messages = [{"role": "user", "content": email.prompt()}]

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=system,
                tools=tools,
                thinking={"type": "adaptive"},
                messages=messages,
            )
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens

            if response.stop_reason != "tool_use":
                why = {
                    "end_turn": "Claude finished without requesting another tool",
                    "refusal": "Claude declined to continue",
                    "max_tokens": "the reply hit the length limit",
                }.get(response.stop_reason, f"stop reason '{response.stop_reason}'")
                text = " ".join(b.text for b in response.content if b.type == "text").strip()
                if text:
                    print(f"\nClaude: {preview(text)}")
                print(f"\nStopped: {why}.")
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            limit_hit = False
            for block in (b for b in response.content if b.type == "tool_use"):
                if tool_calls >= MAX_TOOL_CALLS:
                    limit_hit = True
                    break
                tool_calls += 1
                result, is_error = await call_tool(mcp, block, email, drafts, saved)
                if block.name == "save_reply" and not is_error:
                    saved = dict(block.input)
                print(f"\nStep {tool_calls}: {block.name}")
                print(f"  Inputs: {json.dumps(block.input, ensure_ascii=False)}")
                print(f"  Result{' (error)' if is_error else ''}: {preview(result)}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error})
            if limit_hit or tool_calls >= MAX_TOOL_CALLS:
                print(f"\nStopped: reached the limit of {MAX_TOOL_CALLS} tool calls.")
                break
            messages.append({"role": "user", "content": results})

    if saved:
        tag = REVIEW_TAG if saved["needs_human_review"] else "no HR review flag"
        print(f"\n--- Draft ({tag}) ---\nTo: {email.from_header}\n"
              f"Subject: {email.reply_subject(saved['needs_human_review'])}\n\n{saved['reply_text'].strip()}\n--- End of draft ---")
    else:
        print("\nNo draft was saved for this email.")
    print(f"\nTokens: {tokens_in:,} input, {tokens_out:,} output")
    return tokens_in, tokens_out


async def call_tool(mcp: Client, block, email: Email, drafts, saved: dict | None) -> tuple[str, bool]:
    """Run one tool request. Returns (result text, is_error)."""
    if block.name == "save_reply":
        args = block.input
        if saved is not None:
            return "A reply was already saved for this email. save_reply can be called only once.", True
        if not isinstance(args.get("reply_text"), str) or not args["reply_text"].strip() \
                or not isinstance(args.get("needs_human_review"), bool):
            return "save_reply needs reply_text (non-empty text) and needs_human_review (true or false).", True
        try:
            return drafts.save(email, args["reply_text"], args["needs_human_review"]), False
        except (Fail, imaplib.IMAP4.error, OSError) as e:
            return f"The draft could not be saved: {e}", True
    result = await mcp.call_tool(block.name, dict(block.input))
    text = "\n".join(getattr(c, "text", "") for c in result.content).strip()
    return text or "(empty result)", bool(result.is_error)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def run_file(client: anthropic.Anthropic, path: Path) -> None:
    if not path.is_file():
        raise Fail(f"Email file not found: {path}")
    await run_agent(client, Email(path.read_bytes(), path.name), FileDrafts())


async def run_inbox(client: anthropic.Anthropic) -> None:
    try:
        address, password = load_inbox.read_login(load_inbox.LOGIN_FILE)
    except SystemExit as e:  # read_login's messages never include the file's contents
        raise Fail(str(e))
    try:
        imap = imaplib.IMAP4_SSL(load_inbox.IMAP_HOST)
    except OSError as e:
        raise Fail(f"Couldn't connect to {load_inbox.IMAP_HOST}: {e.__class__.__name__}.")
    try:
        try:
            imap.login(address, password)
        except imaplib.IMAP4.error:
            raise Fail(f"Gmail rejected the sandbox login. Check the app password in "
                       f"{load_inbox.LOGIN_FILE.name} and that IMAP is turned on.")
        finally:
            del password
        if imap.select("INBOX")[0] != "OK":
            raise Fail("Couldn't open the inbox.")
        drafts = GmailDrafts(imap, address)

        _, data = imap.uid("SEARCH", None, "UNSEEN")
        uids = data[0].split() if data and data[0] else []
        print(f"Signed in to the Gmail sandbox. Unread emails in the inbox: {len(uids)}")  # address not shown, so saved runs stay shareable
        total_in = total_out = handled = 0
        for uid in uids:
            uid = uid.decode()
            # Read without changing its flags, then mark it read before working on it,
            # so it is never handled twice, even if this run fails partway.
            _, fetched = imap.uid("FETCH", uid, "(X-GM-THRID FLAGS BODY.PEEK[])")
            part = next((f for f in fetched if isinstance(f, tuple)), None)
            if part is None or b"\\Seen" in part[0]:
                continue  # read since the search (e.g. by another run)
            imap.uid("STORE", uid, "+FLAGS", r"(\Seen)")
            email = Email(part[1], f"inbox message {uid}")
            thread = re.search(rb"X-GM-THRID (\d+)", part[0])
            drafts.original_uid = uid
            drafts.original_thread = thread.group(1).decode() if thread else None
            tin, tout = await run_agent(client, email, drafts)
            total_in += tin
            total_out += tout
            handled += 1
        print(f"\n{'=' * 70}\nHandled {handled} email(s). Total tokens: {total_in:,} input, {total_out:,} output."
              "\nDrafts only: nothing was sent.")
    finally:
        try:
            imap.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


class Tee:
    """Write everything printed to the terminal and to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare draft replies to Hartwell Foods HR emails. Never sends email.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--email", type=Path, metavar="PATH", help="handle one .eml email file; the draft goes to week-04/drafts/")
    mode.add_argument("--inbox", action="store_true",
                      help="handle each unread email in the Gmail sandbox inbox; drafts are created in Gmail")
    parser.add_argument("--save", type=Path, metavar="PATH", help="also write everything printed to this text file (UTF-8)")
    args = parser.parse_args()

    log = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        log = args.save.open("w", encoding="utf-8")
    out = Tee(sys.stdout, log) if log else sys.stdout
    try:
        with redirect_stdout(out):
            try:
                client = load_client()
                asyncio.run(run_inbox(client) if args.inbox else run_file(client, args.email))
            except Fail as e:
                print(f"Error: {e}")
                raise SystemExit(1)
            except anthropic.AuthenticationError:
                print("Error: the API key was rejected. Check the key file.")
                raise SystemExit(1)
            except anthropic.RateLimitError:
                print("Error: rate limit reached. Wait a minute and try again.")
                raise SystemExit(1)
            except anthropic.APIStatusError as e:
                print(f"Error: the Anthropic API returned an error ({e.status_code}): {e.message}")
                raise SystemExit(1)
            except anthropic.APIConnectionError:
                print("Error: couldn't reach the Anthropic API. Check your internet connection.")
                raise SystemExit(1)
            if args.save:
                print(f"\nSaved this output to {args.save}")
    finally:
        if log:
            log.close()


if __name__ == "__main__":
    main()
