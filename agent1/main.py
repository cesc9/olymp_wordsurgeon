"""
agent1/main.py — Prototype LangGraph agent that edits a .docx via three tools.

Scope (v1): one document (black_holes.docx), three operations: insert, replace, remove, all on 
EXACT text matches, with surgical run-level editing that preserves formatting of everything it 
doesn't touch.

Note that the live "Document" object lives in the graph state. Tools receive it via InjectedState
(the LLM never handles the document itself, it only supplies the edit arguments). Edits happen in
place. The document is saved once at the end.

Example Run:
    python agent1/main.py "Replace 'black hole' with 'gravitational abyss'"

Needs ANTHROPIC_API_KEY in .env and Anthropic API credits (makes real calls).
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Iterator, Optional

from dotenv import load_dotenv
from docx import Document
from docx.text.paragraph import Paragraph

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

# Documents live in Word_Docs/ at the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
WORD_DOCS_DIR = REPO_ROOT / "Word_Docs"
INPUT_PATH = WORD_DOCS_DIR / "black_holes.docx"
OUTPUT_PATH = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_edited.docx")


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

TOOLS = [replace_text, remove_text, insert_text]


# ============================================================================
# Graph
# ============================================================================

llm = ChatAnthropic(
    model=MODEL, 
    api_key=API_KEY, 
    max_tokens=2048)

# parallel_tool_calls=False forces one tool call per turn
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

def agent(state: EditState):
    """Ask the LLM what to do next, given the conversation so far."""
    response = llm_with_tools.invoke(state["messages"])
    # print(f'The response looks like this:')
    # print(response)
    return {"messages": [response]}

def should_continue(state: EditState):
    """Route: if the LLM requested a tool, run it, otherwise stop."""
    return "tools" if state["messages"][-1].tool_calls else END


# EditState is the shape of the "whiteboard" (the single data structure the graph carries from node 
# to node). By passing it to StateGraph(...), I am telling LangGraph: "the state of this graph has
# these fields: messages, doc, doc_path."
def build_app():
    g = StateGraph(EditState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue)
    g.add_edge("tools", "agent")
    return g.compile()


# ============================================================================
# Entry point
# ============================================================================

SYSTEM_PROMPT = """You edit a Microsoft Word document using the provided tools.

Rules:
- Make ONLY the change the user requests. Do not touch anything else.
- When calling a tool, pass the EXACT text as it appears in the document (match punctuation and
  capitalization exactly) so the edit lands precisely.
- If the requested text does not appear in the document, say so instead of guessing.
- Use replace_text to change text, remove_text to delete text, and insert_text to add text relative
  to an existing anchor phrase.

Here is the current document content:
-------------------- DOCUMENT --------------------
{document_text}
--------------------------------------------------
"""

# Turns the whole document into a single plain-text string. The text the LLM is shown in the system
# prompt so it knows what's in the document.
def document_text(doc: Document) -> str:
    return "\n".join(p.text for p in iter_all_paragraphs(doc) if p.text.strip())


def main():

    # Reads instruction or falls back to default if no instructions given
    instruction = sys.argv[1] if len(sys.argv) > 1 else (
        "Replace 'black hole' with 'gravitational abyss'"
    )

    if not INPUT_PATH.exists():
        raise SystemExit(f"Document not found: {INPUT_PATH}. Add it to the Word_Docs/ folder.")

    doc = Document(str(INPUT_PATH)) # loads the file from disk into memory as a python-docx object
    app = build_app() # assambles and compiles graph

    # Note that the tools need the real object so we give pass doc but the LLM needs only the text 
    # so we pass document_text(doc)
    app.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(document_text=document_text(doc))},
            {"role": "user", "content": instruction},
        ],
        "doc": doc,
        "doc_path": str(INPUT_PATH),
    })

    doc.save(str(OUTPUT_PATH))
    print(f"Instruction: {instruction}")
    print(f"Saved edited document -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
