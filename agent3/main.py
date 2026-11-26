"""
agent3/main.py — Interactive LangGraph agent that edits a .docx, with formatting, undo and memory.

Builds on agent2 (same interactive document picker from Word_Docs/, same multi-edit loop, same
surgical run-level editing engine). Adds:
  - Formatting (adjust_formatting): set bold/italic/underline, font size, typeface and yellow
    highlight on specific text or the whole document. Splitting runs keeps these edits surgical.
  - Undo: type "undo" to revert the last change; multi-level, via full document snapshots.
  - Short-term memory: the last few instructions AND the agent's replies are passed back as context,
    so follow-ups like "also make it bigger" or "do option 1" work.
  - Robustness: graceful handling of errors and Ctrl-C/Ctrl-D; honest "no change" reporting (only
    saves when the document actually changed); and parallel_tool_calls=False so the shared Document
    is never mutated by concurrent tool calls (which otherwise segfaults via lxml).

The live "Document" object lives in the graph state. Tools receive it via InjectedState (the LLM
never handles the document itself, it only supplies the edit arguments). Edits happen in place.

Run:
    python agent3/main.py or python main.py (within working dir)

Needs ANTHROPIC_API_KEY in .env and Anthropic API credits (makes real calls).
"""

import os
from collections import deque
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Annotated, Iterator, Optional

from dotenv import load_dotenv
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, InjectedState


# ============================================================================
# Get API key and initialize model
# ============================================================================

load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise SystemExit("ANTHROPIC_API_KEY not found. Check your .env file.")

MODEL = "claude-haiku-5-5"

# Documents live in Word_Docs/ at the repo root. Anchor the path to THIS file's location (not the
# current working directory) 
REPO_ROOT = Path(__file__).resolve().parent.parent
WORD_DOCS_DIR = REPO_ROOT / "Word_Docs"
MAX_ATTEMPTS = 3  # how many tries the user gets to name a valid document


# ============================================================================
# Helper function for word doc selection
# ============================================================================

def available_doc_names() -> list[str]:
    """
    Sorted list of document names (without the .docx extension) in Word_Docs/.
    """
    if not WORD_DOCS_DIR.is_dir():
        return []
    return sorted(p.stem for p in WORD_DOCS_DIR.glob("*.docx") if not p.name.startswith("~$"))


def prompt_for_document() -> Path:
    """
    Ask the user which document to edit, validating against Word_Docs/.

    The user gets MAX_ATTEMPTS tries. On an invalid name we list the valid ones and ask again; after
    the final invalid attempt we exit the program cleanly.

    Returns the chosen document's path. Accepts the name with or without `.docx`.
    """
    names = available_doc_names()
    if not names:
        raise SystemExit(f"No .docx files found in {WORD_DOCS_DIR}. Add a document and try again.")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = input("Which word document would you like to edit? "
                    "(Type in the name, e.g. black_holes)\n> ").strip()
        # Normalize: ignore a trailing .docx and surrounding whitespace.
        choice = raw[:-5] if raw.lower().endswith(".docx") else raw

        if choice in names:
            return WORD_DOCS_DIR / f"{choice}.docx"

        # Invalid name, user can try again
        if attempt < MAX_ATTEMPTS:
            print(f"Invalid document name. Possible document names are: {', '.join(names)}")
        else:
            raise SystemExit("Add your desired document to the Word_Docs/ folder "
                             "and try again after.")


# ============================================================================
# Document helpers
# ============================================================================

def iter_all_paragraphs(doc: Document) -> Iterator[Paragraph]:
    """
    Generator to iterate through the whole word document properly (including tables)

    Editing by iterating paragraphs (never rebuilding the doc) is what keeps images, tables, styles,
    and untouched text intact.
    """
    for p in doc.paragraphs:
        yield p
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


def _edit_paragraph_once(
    paragraph: Paragraph, old: str, new: str, search_from: int = 0
) -> int:
    """
    Find the first occurrence of "old" at/after "search_from" in this paragraph's flattened text and
    replace it with "new". Edits RUNS surgically.

    Core routine that solves the run-vs-paragraph problem:
     - We read across runs (paragraph text is the concatenation of run texts), so we can match text
       that spans multiple runs.
     - We write only into the affected runs, leaving every other run and all formatting boundaries
       untouched. "new" inherits the formatting of the run where the match starts.

    Returns the index just after the inserted "new" (where to resume searching), or -1 if `old` was
    not found.
    """
    runs = paragraph.runs
    full_text = "".join(r.text for r in runs)

    start = full_text.find(old, search_from)

    # -1 is returned immediately if there is no (more) "old" found in the paragraph
    if start == -1:
        return -1
    end = start + len(old)  # exclusive

    pos = 0  # running char offset of each run within the flattened text

    # Loop actually modifies the runs in the word doc (does the actual surgery)
    for run in runs:
        run_start = pos
        run_end = pos + len(run.text)
        pos = run_end

        overlap_start = max(run_start, start)
        overlap_end = min(run_end, end)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - run_start
        local_end = overlap_end - run_start
        before = run.text[:local_start]
        after = run.text[local_end:]

        if run_start <= start < run_end:
            # Run that contains the match START -> drop the matched slice, put "new" here.
            run.text = before + new + after
        else:
            # Run fully/partly inside the match (but not the start) → just delete the slice.
            run.text = before + after

    # returns idx immediately after replacement (after inserted "new")
    return start + len(new)


# Document-wide find-and-replace loop. Iterate through all paragraphs and within each paragraph
# keeps replacing until there that paragraph is exhausted (it does not touch runs itself)
def edit_text(doc: Document, old: str, new: str, max_count: Optional[int] = None) -> int:
    """
    Replace "old" with "new" everywhere (or up to "max_count" times) across the whole document
    surgically. " new="" " removes the "old" part.

    Returns the number of edits made.
    """
    if not old:
        return 0
    count = 0

    # Iterate through all paragraphs (complete document text including tables)
    for paragraph in iter_all_paragraphs(doc):

        # We are now within a certain paragraph
        resume = 0
        while True:
            # break if max_count is hit
            if max_count is not None and count >= max_count:
                break

            # Real work, delegated to the helper. _edit_paragraph_once finds the first occurrence of
            # "old" at or after position "resume", replaces it surgically (editing runs, preserving
            # formatting), returns the position just past the newly inserted "new", i.e. where to
            # continue searching next, or -1 if it found nothing. This is where runs are iterated
            # over
            resume = _edit_paragraph_once(paragraph, old, new, resume)

            # -1 is returned by _edit_paragraph_once only if there is no more "old" found in the
            # paragraph (we can then go on to the next paragraph, therefore break)
            if resume == -1:
                break  # no more matches in this paragraph (breaks while loop)
            count += 1
    return count


# ----------------------------------------------------------------------------
# Formatting engine (bold / italic / underline), run-splitting
# ----------------------------------------------------------------------------
"""
The core idea: "run-splitting" 

This is the key concept, and it comes straight from how Word stores formatting. Formatting isn't
attached to individual characters. It's attached to a run, and a run is an all-or-nothing block of
uniformly-styled text. You cannot say "the 3rd to 5th character of this run is bold." The run is the
smallest unit that can carry a style. A run is either entirely bold or entirely not. That creates a 
problem: what if you want to bold just one word inside a run that contains a whole sentence? The 
sentence is one run, and you can't bold "part of a run."

The solution is run-splitting: before you can style part of a run, you physically cut the run into
pieces so that the text you want to style becomes a run of its own. Then you style that whole new 
run, which is now allowed, because it's an entire run.

So "run-splitting" is the enabling trick of the whole formatting engine. The sequence is always:

  1. Find the text you want to style.
  2. Split the run(s) so that text stands alone as its own run(s).
  3. Style that run.
  4. Everything else is untouched because you only ever cut at the edges of the target.

run-splitting is what lets formatting be surgical. You carve out exactly the target, style it, and
every other run comes out byte-for-byte identical.

_split_run — the cutter
Cuts one run into two at a given spot, copying the formatting onto both halves so nothing changes 
visually. This is the raw "scissors." It's the only function that actually touches the XML to divide
a run. Knows nothing about paragraphs just "cut this run here."

_split_run_at — the aimer
Takes a paragraph position ("I want a boundary at character N") and figures out which run that 
position lands in, then calls _split_run to make the cut. It translates "position in the whole
paragraph" into "this run, this local offset." The scissors need to be told exactly where, this is
what points them.

_apply_run_attrs — the styler
Takes one run and sets the requested attributes (bold/italic/underline/size/font/highlight), 
leaving any attribute passed as None untouched. This is the only function that actually applies 
styling. Both top-level tools funnel through it, so "bold" means the same thing everywhere.

format_text — the targeted tool (orchestrates the three above)
"Style every occurrence of this substring." For each match it:

finds the match's start/end positions,
calls _split_run_at at both edges → the match becomes its own run(s),
calls _apply_run_attrs on the runs that fall inside the match.
This is the function that does surgical, word-level formatting.

format_document — the whole-document tool
"Style everything" (e.g. make the whole doc Arial). No searching, no cutting needed, 
every run is styled in full.
"""


# Formatting (bold, italic, ...) lives on a RUN, and a run has ONE uniform formatting. To style just
# part of a run we must first SPLIT the run so the target text occupies whole run(s) of its own, 
# then flip the flags on those.
def _split_run(run: Run, k: int) -> Run:
    """
    Split "run" at local offset k: characters [:k] stay in "run", characters [k:] move to a NEW run
    inserted immediately after. The new run carries the SAME formatting (we copy the run's <w:rPr>),
    so appearance is unchanged until we deliberately restyle it. Returns the new run.
    """

    text = run.text
    new_r = OxmlElement("w:r")
    rpr = run._r.find(qn("w:rPr")) # the run's formatting properties, if any
    if rpr is not None:
        new_r.append(deepcopy(rpr)) # copy formatting onto the new run
    run._r.addnext(new_r) # insert the new run right after this one
    new_run = Run(new_r, run._parent)
    new_run.text = text[k:]
    run.text = text[:k]
    return new_run


def _split_run_at(paragraph: Paragraph, char_pos: int) -> None:
    """
    Ensure "char_pos" falls on a run boundary by splitting the straddling run (if any). After 
    calling this for both ends of a range, every run is either fully inside or fully outside that
    range.
    """
    pos = 0
    for run in paragraph.runs:
        run_start, run_end = pos, pos + len(run.text)
        if run_start < char_pos < run_end:   # strictly inside this run → split
            _split_run(run, char_pos - run_start)
            return
        pos = run_end


def _apply_run_attrs(
    run: Run,
    bold: Optional[bool],
    italic: Optional[bool],
    underline: Optional[bool],
    size_pt: Optional[float],
    font_name: Optional[str],
    highlight: Optional[bool],
) -> None:
    """
    Apply the requested formatting attributes to a single run. A None attribute is left unchanged.
    Shared by targeted (`format_text`) and whole-document (`format_document`) formatting so both 
    behave identically.
    """
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if underline is not None:
        run.underline = underline
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if font_name is not None:
        run.font.name = font_name
    if highlight is not None:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW if highlight else WD_COLOR_INDEX.AUTO


def format_text(
    doc: Document,
    target: str,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    underline: Optional[bool] = None,
    size_pt: Optional[float] = None,
    font_name: Optional[str] = None,
    highlight: Optional[bool] = None,
) -> int:
    """
    Set bold/italic/underline/size/font/highlight on every exact occurrence of "target".

    For each match we make its start and end fall on run boundaries (splitting runs
    as needed), then apply only the requested attributes to the runs that lie fully
    inside the match. An attribute of None is left unchanged; bold/italic/underline
    True turns on and False turns off; "size_pt" sets the font size in points;
    `font_name` sets the typeface (e.g. "Arial"); `highlight` True applies a YELLOW
    text highlight and False removes any highlight (yellow is the only supported color).
    Text content and every other run are preserved. Returns the number of matches styled.
    """
    if not target or all(v is None for v in (bold, italic, underline, 
                                             size_pt, font_name, highlight)):
        return 0
    count = 0
    for paragraph in iter_all_paragraphs(doc):
        resume = 0
        while True:
            full_text = "".join(r.text for r in paragraph.runs)
            start = full_text.find(target, resume)
            if start == -1:
                break
            end = start + len(target)

            # Make the match occupy whole runs, then style those runs.
            _split_run_at(paragraph, start)
            _split_run_at(paragraph, end)
            pos = 0
            for run in paragraph.runs:
                run_start, run_end = pos, pos + len(run.text)
                pos = run_end
                if run_end > run_start and start <= run_start and run_end <= end:
                    _apply_run_attrs(run, bold, italic, underline, size_pt, font_name, highlight)

            count += 1
            resume = end  # continue past this match (target is non-empty → always advances)
    return count


def format_document(
    doc: Document,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    underline: Optional[bool] = None,
    size_pt: Optional[float] = None,
    font_name: Optional[str] = None,
    highlight: Optional[bool] = None,
) -> int:
    """Apply formatting to EVERY run in the whole document (body + table cells).

    This is the "everything" scope: no text matching or run-splitting is needed because
    we style entire runs. A None attribute is left unchanged. Returns the number of
    (non-empty) runs styled.
    """
    if all(v is None for v in (bold, italic, underline, size_pt, font_name, highlight)):
        return 0
    count = 0
    for paragraph in iter_all_paragraphs(doc):
        for run in paragraph.runs:
            if run.text:
                _apply_run_attrs(run, bold, italic, underline, size_pt, font_name, highlight)
                count += 1
    return count


# ============================================================================
# State
# ============================================================================

class EditState(MessagesState):
    """MessagesState (messages + add_messages reducer) plus the live document doc and doc_path."""
    doc: Document
    doc_path: str


# ============================================================================
# Tools
# ============================================================================

@tool
def replace_text(old: str, new: str, state: Annotated[dict, InjectedState]) -> str:
    """Replace every exact occurrence of "old" with "new". Use the exact text as it appears in the
    document. Formatting of untouched text is preserved."""
    n = edit_text(state["doc"], old, new)
    if n == 0:
        return f"No occurrence of {old!r} found. Nothing changed."
    return f"Replaced {n} occurrence(s) of {old!r} with {new!r}."


@tool
def remove_text(text: str, state: Annotated[dict, InjectedState]) -> str:
    """Remove every exact occurrence of "text" from the document. Use the exact text as it appears.
    Formatting of untouched text is preserved."""

    # Essentially does exactly what replace_text is doing but it replaces with "" (= removes it)
    n = edit_text(state["doc"], text, "")
    if n == 0:
        return f"No occurrence of {text!r} found. Nothing changed."
    return f"Removed {n} occurrence(s) of {text!r}."


@tool
def insert_text(
    anchor: str, text: str, state: Annotated[dict, InjectedState], position: str = "after"
) -> str:
    """Insert "text" relative to the FIRST occurrence of "anchor". position="after" (default)
    inserts right after the anchor; "before" inserts right before it. The anchor itself is kept.
    Inserted text inherits the anchor's formatting."""
    if position not in ("after", "before"):
        return f"Invalid position {position!r}; use 'after' or 'before'."
    replacement = anchor + text if position == "after" else text + anchor
    n = edit_text(state["doc"], anchor, replacement, max_count=1)
    if n == 0:
        return f"Anchor {anchor!r} not found. Nothing inserted."
    return f"Inserted text {position} {anchor!r}."


@tool
def adjust_formatting(
    state: Annotated[dict, InjectedState],
    target: str = "",
    whole_document: bool = False,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    underline: Optional[bool] = None,
    size_pt: Optional[float] = None,
    font_name: Optional[str] = None,
    highlight: Optional[bool] = None,
    requested_color: Optional[str] = None,
) -> str:
    """
    Set bold/italic/underline, font size, typeface, and/or highlight on text.

    Choose a scope:
    - Set "target" to format every exact occurrence of that text (pass it exactly as it
      appears in the document).
    - Set "whole_document=true" to format the ENTIRE document at once (e.g. "put everything
      in Arial"); `target` is then ignored.

    For bold, italic, underline, highlight: true turns it ON, false turns it OFF, and
    omitting it leaves that attribute unchanged. `size_pt` sets the font size in points
    (e.g. 14); `font_name` sets the typeface (e.g. "Arial", "Times New Roman"). Highlighting
    is only supported in YELLOW: if the user asks to highlight in a specific color, set
    highlight=true and pass that color as `requested_color` (e.g. "green") so the result can
    note that yellow was used instead. Only formatting changes. Text and all other content
    are preserved. This does NOT insert text: to style text you are adding, call insert_text
    first, then adjust_formatting on the inserted text.
    """

    if all(v is None for v in (bold, italic, underline, size_pt, font_name, highlight)):
        return ("No formatting specified. Set at least one of bold, italic, underline, "
                "size_pt, font_name, or highlight.")

    if whole_document:
        n = format_document(state["doc"], bold=bold, italic=italic, underline=underline,
                            size_pt=size_pt, font_name=font_name, highlight=highlight)
        scope = "the entire document"
        none_found = n == 0  # only if the document has no text runs at all
    else:
        if not target:
            return "Specify `target` text to format, or set whole_document=true."
        n = format_text(state["doc"], target, bold=bold, italic=italic, underline=underline,
                        size_pt=size_pt, font_name=font_name, highlight=highlight)
        scope = f"{n} occurrence(s) of {target!r}"
        none_found = n == 0

    if none_found:
        if whole_document:
            return "The document has no editable text. Nothing changed."
        return f"No occurrence of {target!r} found. Nothing changed."

    applied = ", ".join(
        f"{name}={val}"
        for name, val in (("bold", bold), ("italic", italic), ("underline", underline),
                          ("size_pt", size_pt), ("font_name", font_name), ("highlight", highlight))
        if val is not None
    )
    msg = f"Set formatting ({applied}) on {scope}."
    # Be transparent when we substituted yellow for a requested color.
    if highlight and requested_color and requested_color.strip().lower() not in ("yellow", ""):
        msg += (f" Note: only yellow highlighting is supported, so it was highlighted in YELLOW "
                f"instead of the requested color {requested_color!r}.")
    return msg


TOOLS = [replace_text, remove_text, insert_text, adjust_formatting]


# ============================================================================
# Graph
# ============================================================================

llm = ChatAnthropic(
    model=MODEL, 
    api_key=API_KEY, 
    max_tokens=2048)

# parallel_tool_calls=False forces the model to emit AT MOST ONE tool call per turn.
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)


def agent(state: EditState):
    """Ask the LLM what to do next, given the conversation so far."""
    response = llm_with_tools.invoke(state["messages"])
    # print(f'The response looks like this:')
    # print(response)
    return {"messages": [response]}


def should_continue(state: EditState):
    """Route: if the LLM requested a tool, run it, otherwise stop."""
    # This edge links the agent node with the tools node (function returns tools if we should go to
    # the tools node next and else it returns END if we should go to the END node next)
    return "tools" if state["messages"][-1].tool_calls else END


def build_app():
    g = StateGraph(EditState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    # This conditional edge links the agent node with the tools node (function returns tools if we 
    # should go to the tools node next and else it returns END if we should go to the END node next)
    g.add_conditional_edges("agent", should_continue)
    g.add_edge("tools", "agent")
    return g.compile()


# ============================================================================
# Entry point
# ============================================================================

SYSTEM_PROMPT = """You edit a Microsoft Word document using the provided tools.

Rules:
- Make ONLY the change the user requests. Do not touch anything else.
- When calling a tool, pass the EXACT text as it appears in the document
  (match punctuation and capitalization exactly) so the edit lands precisely.
- If the requested text does not appear in the document, say so instead of guessing.
- Use replace_text to change text, remove_text to delete text, and insert_text to add
  text relative to an existing anchor phrase.
- Use adjust_formatting to make existing text bold, italic, and/or underlined (or to turn
  those off), to set its font size in points, to change its typeface (font name), and/or to
  highlight it. Scope it either to specific text (set `target`) or to the whole document (set
  `whole_document=true`, e.g. "put everything in Arial"). Highlighting is only supported in
  YELLOW; if the user asks for another color, still highlight (set highlight=true) and pass
  their color as requested_color so the result notes that yellow was used instead.
  adjust_formatting only changes formatting, never the text. To add NEW styled text, first
  insert_text, then call adjust_formatting on the exact text you inserted.

{recent_context}
Here is the current document content:
-------------------- DOCUMENT --------------------
{document_text}
--------------------------------------------------
"""


# Turns the whole document into a single plain-text string. The text the LLM is shown in the system
# prompt so it knows what's in the document.
def document_text(doc: Document) -> str:
    return "\n".join(p.text for p in iter_all_paragraphs(doc) if p.text.strip())


def recent_context(recent: "deque[tuple[str, str]]", reply_chars: int = 300) -> str:
    """
    Render the last few (user instruction, agent reply) exchanges as context, so the
    model can resolve follow-up references ("also…", "make it bigger", "do option 1",
    "undo that" etc.) that build on earlier turns. We include the agent's replies (truncated)
    so references to what it previously said/did can be resolved. This is PAST context only;
    the NEW instruction arrives as the user message, and the CURRENT document is always shown
    fresh below, so a stale past reply never overrides the live document.
    """
    
    if not recent:
        return ""  # first instruction of the session — nothing to add
    blocks = []
    for i, (user, reply) in enumerate(recent, start=1):
        short = reply.strip().replace("\n", " ")
        if len(short) > reply_chars:
            short = short[:reply_chars].rstrip() + "…"
        blocks.append(f"  {i}. User: {user}\n     You: {short}")
    lines = "\n".join(blocks)
    return (
        "For context, the recent exchanges (oldest to newest) were:\n"
        f"{lines}\n"
        "Use these only to interpret follow-up references in the NEW instruction; act on the "
        "NEW instruction, not on these past ones.\n"
    )

# For comparison purposes only
def doc_signature(doc: Document) -> list:
    """
    A snapshot of text AND run formatting across the whole document.

    Used to detect whether an instruction actually changed anything. We include bold/italic/
    underline so a formatting-only edit (which leaves the visible text identical) is still
    recognised as a change and gets saved.
    """
    return [
        (run.text, run.bold, run.italic, run.underline, run.font.size,
         run.font.name, run.font.highlight_color)
        for para in iter_all_paragraphs(doc)
        for run in para.runs
    ]


QUIT_WORDS = {"quit", "exit", "q"}
UNDO_WORDS = {"undo", "revert"}


# For restoration purposes
def snapshot(doc: Document) -> bytes:
    """
    Serialise the whole document to in-memory bytes, a complete, restorable snapshot of its current
    state, (text, formatting, images, tables, everything). Used for undo: reloading these bytes with
    Document(BytesIO(...)) reproduces the document exactly, so reverting restores a known-good state
    rather than trying to compute the inverse of an edit.
    """
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def final_reply_text(result) -> str:
    """
    Pull the agent's final plain-text answer out of the returned state.

    The last message's ".content" is usually a string, but with extended thinking it can be a list 
    of blocks (both cases are handled here).
    """
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    # List of blocks: keep only the text ones.
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def main():

    # Ask the user which document to edit (validated against Word_Docs/).
    input_path = prompt_for_document()
    # Edited result is written next to the original as "<name>_edited.docx".
    output_path = input_path.with_name(f"{input_path.stem}_edited.docx")

    doc = Document(str(input_path))  
    app = build_app()

    print(f"\nEditing '{input_path.name}'. "
          "Describe a change, or 'q' to finish.\n")

    edits_applied = 0 # how many instructions ACTUALLY changed the document
    history: list[bytes] = [] # snapshots of prior states; one per applied change (undo stack)
    recent: deque[tuple[str, str]] = deque(maxlen=3) # last few (instruction, reply) exchanges
    
    while True:

        # Reading input can raise EOFError (piped input ends / Ctrl-D) or KeyboardInterrupt 
        # (Ctrl-C). Treat both as a graceful quit, not a crash.
        try:
            instruction = input(
                "What change would you like to make? (type 'undo' to revert, 'q' to finish)\n> "
            ).strip()
        # Exits smoothly if user does a keyboard interrupt
        except (EOFError, KeyboardInterrupt):
            print()
            break


        # Ignore empty input, ask again
        if not instruction:
            continue
        if instruction.lower() in QUIT_WORDS:
            break

        # Undo: pop the most recent snapshot and restore it (no API call needed).
        if instruction.lower() in UNDO_WORDS:
            if not history:
                print("Nothing to undo.\n")
                continue
            doc = Document(BytesIO(history.pop())) # Replace the live doc with the prior state
            doc.save(str(output_path))
            edits_applied = max(0, edits_applied - 1)
            recent.append((instruction, "Reverted the last change."))  # keep memory coherent
            print(f"Reverted the last change. ({len(history)} undo step(s) remaining)")
            print(f"(saved -> {output_path})\n")
            continue

        # Snapshot the document (text + formatting) so we can tell whether this instruction actually
        # changed anything (rather than assuming every turn edits).
        before = doc_signature(doc)
        # Full byte snapshot of the pre-edit state, kept so this change can be undone.
        before_bytes = snapshot(doc)

        # An edit run makes a live API call and can fail (no credits, network, rate limit, a
        # malformed response). Catch it, tell the user, and keep the session alive so earlier edits
        # aren't lost and they can try again.
        try:
            # Run the agent on the CURRENT document state. System prompt is rebuilt each turn so the
            # model sees the latest text (including earlier edits), and keep mutating the SAME `doc`
            # so changes accumulate.
            result = app.invoke({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(
                        recent_context=recent_context(recent),
                        document_text=document_text(doc),
                    )},
                    {"role": "user", "content": instruction},
                ],
                "doc": doc,
                "doc_path": str(input_path),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  That edit could not be completed: {type(exc).__name__}: {exc}")
            print("   Nothing was changed. You can try again or rephrase.\n")
            continue

        # Always show the agent's own summary of what it did
        reply = final_reply_text(result)
        print(f"\n{reply}")

        # Record this exchange (instruction + reply) as context for the next turn's follow-ups.
        recent.append((instruction, reply))

        # Only save + count when the document actually changed. If the target text wasn't found (or
        # the model only asked a clarifying question), the document is untouched.
        if doc_signature(doc) != before:
            history.append(before_bytes) # record the pre-edit state so this change can be undone
            doc.save(str(output_path))
            edits_applied += 1
            print(f"(saved -> {output_path})  —  type 'undo' to revert\n")
        else:
            print("(no change made to the document — nothing was saved)\n")

    # Closing summary
    if edits_applied:
        print(f"\nApplied {edits_applied} change(s). Final document saved to {output_path}.")
    else:
        print(f"\nNo changes made. '{input_path.name}' left untouched.")
    print("\nHave a nice day!")


if __name__ == "__main__":
    main()
