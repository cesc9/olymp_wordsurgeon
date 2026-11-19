# olymp_wordsurgeon

An AI-powered Microsoft Word (`.docx`) editor built with **Python** and **LangGraph**.
You give it a natural-language instruction ("replace X with Y", "remove this sentence",
"insert this after that phrase") and an agent carries it out by calling purpose-built
editing tools, editing the document **surgically** so that everything it isn't asked to
change (styled text, tables, images, unrelated paragraphs) stays byte-for-byte intact.

The project is developed as three incremental versions. Each builds directly on the one
before it:

| Version | Folder | Adds |
|---|---|---|
| **Agent 1** | `agent1/` | The core LangGraph agent + surgical editing engine (one-shot CLI) |
| **Agent 2** | `agent2/` | Interactive document selection + a multi-edit REPL loop |
| **Agent 3** | `agent3/` | Formatting tool, whole-document scope, undo, short-term memory, robustness (error handling, honest "no change" reporting, safe sequential tool execution) |

> Each agent is a self-contained `main.py`. Agent 2 is Agent 1 plus the Chapter 2 changes;
> Agent 3 is Agent 2 plus the Chapter 3 changes. Nothing below the editing engine changes
> between versions, only the entry point (`main()`) and its helpers evolve.

---

## Running

### 1. Set up the environment and install dependencies

The project runs in a conda environment named `wordsurgeon`. On macOS (with
[Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed), from the
repo root:

```bash
# create and activate the environment (Python 3.13 is the supported baseline)
conda create -n wordsurgeon python=3.13
conda activate wordsurgeon

# install all dependencies
pip install -r requirements.txt
```

The agent calls the **Anthropic API**, so it needs an API key. Create a `.env` file at the
repo root with your key (this file is git-ignored and must never be committed):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

> The key is loaded at runtime via `python-dotenv`. Billing is on the Anthropic **Console**
> account (separate from any Claude.ai subscription). A `400 "credit balance too low"` error
> means that account needs credits (it is not a code bug).

### 2. Add the documents you want to edit

Place the Word documents (`.docx`) you want to edit in the **`Word_Docs/`** folder. The agent
only lists and edits documents from there. Three sample documents are already included
(`black_holes.docx`, `general_relativity.docx`, `string_theory.docx`).

Edited output is written **next to the original** as `<name>_edited.docx`, leaving your input
file untouched.

### 3. Start the agent and make edits

The main agent in this repo is `agent3/main.py`. Start the agent with:

```bash
python agent3/main.py
```

It will ask which document in `Word_Docs/` you want to edit, then repeatedly prompt
*"What change would you like to make?"*. Type natural-language instructions such as:

```
Replace 'black hole' with 'gravitational abyss'
Remove the sentence about Einstein
Insert 'This is a draft.' before the first paragraph
Make every occurrence of 'singularity' bold
```

Type `undo` (or `revert`) to step back the last applied change, and `quit` / `exit` / `q`
when you're done.

In full, `agent3` can:

- **Replace text** — swap every exact occurrence of a phrase for another.
- **Remove text** — delete every exact occurrence of a phrase.
- **Insert text** — add text immediately before or after an anchor phrase.
- **Format text** — set **bold**, **italic**, **underline**, **font size** (points), **typeface**
  (e.g. Arial), or a **yellow highlight** on a target phrase or the whole document, without changing
  the text itself.
- **Undo** — step back through applied changes, all the way to the originally loaded document.
- **Do several edits from one instruction** — a single request can trigger multiple edits at once
  (e.g. "make every heading bold and replace X with Y"); the agent chains the needed tool calls.
- **Take instructions one after another** — the interactive loop keeps running, so you edit the same
  document across as many turns as you like, with each change building on the last.
- **Remember recent turns** — it keeps short-term memory of the last few exchanges, so follow-ups that
  refer back ("make it bigger", "do that to the next heading too", "undo that") are understood.
- **Talk back to you** — when an instruction is ambiguous or the target text isn't found, the agent
  replies in natural language (asking a clarifying question or reporting what it did) instead of
  guessing and making a wrong edit.

Edits are **surgical**: only the text the instruction names is touched, and surrounding styled runs,
tables, and images stay intact. See [Chapter 3](#chapter-3--agent-3-formatting-undo-memory-and-robustness)
for the design details, and [Limitations](#limitations-agent3) for what it can't do.

> **Other agents.** `agent1/main.py` is a one-shot variant that takes the instruction as a
> command-line argument (`python agent1/main.py "Replace 'black hole' with 'gravitational
> abyss'"`); `agent2/main.py` is the interactive version without Agent 3's formatting, undo
> and robustness fixes. `agent3/main.py` is the recommended entry point.

---

## Chapter 1 — Agent 1: the base LangGraph agent

Agent 1 establishes the whole architecture. Everything in later versions sits on top of this.

### The graph: an agent <-> tools loop

The agent is a standard LangGraph **agent/tools loop**:

```
START → agent → (did the LLM request a tool?)
                   │ yes → tools → back to agent
                   │ no  → END
```

- The **`agent`** node calls the LLM. The LLM decides whether an editing tool is needed and,
  if so, emits one or more tool calls.
- A **conditional edge** (`should_continue`) inspects the LLM's last message: if it contains
  tool calls, route to the `tools` node; otherwise route to `END`.
- The **`tools`** node (a prebuilt `ToolNode`) runs the requested tool(s), then always loops
  back to `agent` so the model can see the results and decide the next step.

The loop keeps cycling until the model stops requesting tools and returns a plain-text reply.
A single instruction can therefore trigger several edits in one run (the model may batch
multiple tool calls, or chain them across iterations).

### State: the document rides alongside the conversation

The graph's shared state extends LangGraph's `MessagesState`:

```python
class EditState(MessagesState):   # inherits `messages` + its add_messages reducer
    doc: Document                 # the live python-docx document being edited
    doc_path: str
```

`messages` carries the conversation; `doc` carries the **live document object**. This
separation is deliberate: the LLM only ever produces small text arguments (the *what* of an
edit). The actual `Document` is never serialized into the conversation — instead the editing
tools reach it directly via LangGraph's `InjectedState`, so the model supplies the edit
arguments while trusted Python code performs the mutation.

### The three editing tools

The LLM is given three tools (it supplies the text arguments; `state` is injected):

- **`replace_text(old, new)`** — replace every exact occurrence of `old` with `new`.
- **`remove_text(text)`** — delete every exact occurrence (this is just `replace_text` with
  an empty replacement).
- **`insert_text(anchor, text, position)`** — insert `text` immediately before/after the first
  occurrence of an existing `anchor` phrase (the anchor itself is kept).

All three are thin wrappers over a single editing engine, so there is only one code path that
can touch a document — fewer ways to corrupt it.

### The editing engine: surgical, run-aware edits

The reliability of the whole project comes from **editing runs, not paragraphs**. In Word, a
visible sentence is stored as multiple *runs*, each with its own formatting; flattening a
paragraph to a string and rewriting it would collapse those runs and destroy formatting. The
engine avoids that:

- **`iter_all_paragraphs(doc)`** yields every paragraph in the body **and** inside table cells,
  so edits never require rebuilding the file (which is what preserves images, styles and tables).
- **`edit_text(...)`** is the coordinator: it walks paragraphs, counts edits, and respects an
  optional `max_count` (used by `insert_text` to affect only the first match).
- **`_edit_paragraph_once(...)`** is the surgeon: it searches across the paragraph's runs so a
  match can span several runs, then edits **only** the runs the match touches. Runs outside the
  match are left completely untouched; replacement text inherits the formatting of the run where
  the match starts.

### Interface

A one-shot CLI: the instruction is passed as a command-line argument (falling back to a default
if none is given). The document path (`black_hole.docx`) is hard-coded, the agent runs once, and
the result is saved to `black_hole_edited.docx`.

---

## Chapter 2 — Agent 2: document selection + interactive editing

Agent 2 keeps the Chapter 1 architecture **unchanged** (same graph, state, tools and editing engine)
and only reworks the entry point, turning the one-shot script into an interactive tool:

1. **Document selection.** Instead of a hard-coded path, it lists the `.docx` files in `Word_Docs/`
   and asks which to edit (name accepted with or without `.docx`; invalid names re-prompt, exiting
   after 3 attempts).
2. **A multi-edit loop.** It then repeatedly prompts *"What change would you like to make?"* until
   you type `quit` / `exit` / `q`. The **same live `doc` is reused** so edits accumulate, and the
   **system prompt is rebuilt each turn** with the current document text so the model always reasons
   about the latest state. After each edit the document is saved to `<name>_edited.docx`.

---

## Chapter 3 — Agent 3: formatting, undo, memory, and robustness

Agent 3 extends Agent 2 with a **formatting** tool, an **undo** command, short-term memory, and several
robustness fixes. The editing engine and graph shape are unchanged.

**1. Formatting (`adjust_formatting`).** A fourth tool changes *how* text looks — **bold**, **italic**,
**underline**, **font size**, **typeface**, **highlight** (yellow only) — on a `target` phrase or the
`whole_document`. Formatting lives on a **run** (one uniform style), so styling part of a run means
splitting it: `_edit_paragraph_once`'s run-splitting is reused so only the runs inside the match change,
while neighbours keep their exact text and formatting.

**2. Undo.** `undo` / `revert` reverts the last applied change, **multi-level** back to the original.
Each applied change pushes a full **byte snapshot** onto a stack that `undo` pops and reloads — restoring
a known-good state is trivially correct and cannot corrupt the document.

**3. Short-term memory.** The last few (instruction, reply) exchanges are kept, so follow-ups that refer
back ("make it bigger", "do that to the next heading", "undo that") are understood; the current document
is always shown fresh, so memory never overrides the live state.

**4. Safe sequential execution.** All tools mutate one live `Document` through lxml, which isn't
thread-safe for concurrent mutation — `ToolNode`'s parallel batched calls caused intermittent segfaults.
Binding the model with `parallel_tool_calls=False` forces **one tool call per turn**.

**5. Robustness & honest reporting.** The session never crashes: **Ctrl-C** / **Ctrl-D** quit gracefully
and a failed API call prints a friendly message and lets you retry. And Agent 3 compares a document
**signature** (text + formatting) before and after each run, so it only saves and counts a turn that
**actually changed** the document, printing `(no change made …)` otherwise.

---

## Limitations (agent3)

Transparent limitations are preferred over unsupported claims. The main ones:

### Size and capacity

1. **Document length.** The document's text is sent to the model on every turn, so the ceiling is the
   model's **context window (~200k tokens, roughly 100k–150k words)**, not RAM. Only text counts, so an
   image- or table-heavy file can be large on disk yet small in tokens.
2. **Edits per instruction.** You can send unlimited instructions per session, but *one* instruction is
   capped by LangGraph's `recursion_limit` (25 steps) at roughly **~12 sequential edits** (handled
   gracefully, not a crash).
3. **Instruction length.** No code limit, but the instruction shares the same ~200k-token budget as the
   document text, a near-maximal document leaves less room for a long instruction.

### Content the tools cannot reach

4. **Hyperlinked text is skipped.** python-docx's `paragraph.runs` excludes runs inside `<w:hyperlink>`
   elements, so the tools silently skip hyperlinked text (common in web-pasted docs). The model still
   *sees* it, so it may try to edit a linked phrase and be told "not found". Treat heavily-hyperlinked
   documents as only partially editable.
5. **Tables: text editable, structure not.** Cell text is fully editable and formatting-safe, but the
   model can't see table structure (no positional edits like "row 2, column 2"; cell text is addressed
   by content), rows/columns can't be added or removed, and nested tables aren't reached.
6. **Images are preserved, not edited.** Images pass through untouched, but the agent can't add, remove,
   move, or replace them — editing operates on text runs only.

### Environment

7. **Python version.** Use **Python ≤ 3.13**; Python 3.15's experimental interpreter showed lxml instability.

### Memory

8. **Bounded short-term memory.** Only the last few exchanges (currently 3) are kept, so follow-ups that
   refer back further than that window are no longer understood. Can be changed easily though.
