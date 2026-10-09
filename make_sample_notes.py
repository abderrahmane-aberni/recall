"""Generates a fictional 5-page lecture-notes PDF used as Recall's preloaded
demo document and as the fixture for the retrieval eval set. Mirrors the
reportlab pattern used for CV-Maxxing's sample_cv.pdf.

Run from the repo root: python make_sample_notes.py
Writes lecture_notes_databases.pdf next to this script (repo root) - that's
where run_eval.py and run_chunk_experiment.py expect to find it.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

OUT = str(Path(__file__).resolve().parent / "lecture_notes_databases.pdf")

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
    title="Introduction to Databases - Lecture Notes",
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, leading=22)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6,
    textColor=colors.HexColor("#1B2430"),
)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=6)
bullet_style = ParagraphStyle("Bullet", parent=body, leftIndent=8)

story = []

# ---------------- Page 1: Course intro + ACID ----------------
story.append(Paragraph("Introduction to Databases", title_style))
story.append(Paragraph("Lecture 4: Transactions and Reliability", styles["Heading3"]))
story.append(Spacer(1, 10))

story.append(Paragraph("The ACID Properties", h2))
story.append(Paragraph(
    "A transaction is a sequence of database operations that is treated as a single "
    "logical unit of work. Relational databases guarantee four properties for every "
    "transaction, known by the acronym ACID.",
    body,
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Atomicity</b> - a transaction either completes entirely or has no effect "
        "at all; there is no partial completion.", bullet_style)),
    ListItem(Paragraph(
        "<b>Consistency</b> - a transaction can only bring the database from one "
        "valid state to another, never violating defined rules such as constraints.",
        bullet_style)),
    ListItem(Paragraph(
        "<b>Isolation</b> - concurrent transactions produce the same result as if "
        "they had run one after another, sequentially.", bullet_style)),
    ListItem(Paragraph(
        "<b>Durability</b> - once a transaction is committed, it remains committed "
        "even in the event of a power loss or crash.", bullet_style)),
], bulletType="bullet", start="circle", leftIndent=14))

story.append(PageBreak())

# ---------------- Page 2: Isolation levels ----------------
story.append(Paragraph("Transaction Isolation Levels", h2))
story.append(Paragraph(
    "The SQL standard defines four isolation levels, each permitting a different "
    "set of concurrency anomalies in exchange for better performance.",
    body,
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Read Uncommitted</b> - transactions can see uncommitted changes made by "
        "other transactions (dirty reads). The weakest and fastest level.", bullet_style)),
    ListItem(Paragraph(
        "<b>Read Committed</b> - a transaction only ever sees data that has been "
        "committed. This is the default isolation level in PostgreSQL.", bullet_style)),
    ListItem(Paragraph(
        "<b>Repeatable Read</b> - if a transaction reads a row twice, it will see the "
        "same data both times, preventing non-repeatable reads.", bullet_style)),
    ListItem(Paragraph(
        "<b>Serializable</b> - the strongest level; concurrent transactions behave as "
        "if they had executed one at a time, in some serial order.", bullet_style)),
], bulletType="bullet", start="circle", leftIndent=14))
story.append(Paragraph(
    "Higher isolation levels reduce anomalies but increase the chance of transactions "
    "blocking each other or being forced to retry after a conflict.",
    body,
))

story.append(PageBreak())

# ---------------- Page 3: Indexing ----------------
story.append(Paragraph("Indexing: B-Trees and Hash Indexes", h2))
story.append(Paragraph(
    "An index is a separate data structure that lets the database find rows without "
    "scanning the entire table. Two common index types are B-tree and hash indexes.",
    body,
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>B-tree indexes</b> keep keys in sorted order, which supports equality "
        "lookups, range queries (e.g. age BETWEEN 20 AND 30), and ORDER BY. "
        "This is the default index type in PostgreSQL and MySQL.", bullet_style)),
    ListItem(Paragraph(
        "<b>Hash indexes</b> store a hash of each key and support only equality "
        "lookups (e.g. WHERE id = 5); they cannot answer range queries because hashing "
        "destroys the original ordering of the values.", bullet_style)),
], bulletType="bullet", start="circle", leftIndent=14))
story.append(Paragraph(
    "Rule of thumb: use a B-tree index unless you know every query against that "
    "column will be an exact-match lookup and you need marginally faster point "
    "lookups than a B-tree provides.",
    body,
))

story.append(PageBreak())

# ---------------- Page 4: Normalization ----------------
story.append(Paragraph("Normal Forms", h2))
story.append(Paragraph(
    "Normalization is the process of structuring tables to reduce data redundancy "
    "and avoid update anomalies.",
    body,
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>First Normal Form (1NF)</b> - every column holds a single, atomic value; "
        "no repeating groups or arrays inside a single column.", bullet_style)),
    ListItem(Paragraph(
        "<b>Second Normal Form (2NF)</b> - the table is in 1NF, and every non-key "
        "column depends on the whole primary key, not just part of it (relevant "
        "mainly for composite keys).", bullet_style)),
    ListItem(Paragraph(
        "<b>Third Normal Form (3NF)</b> - the table is in 2NF, and every non-key "
        "column depends only on the primary key, not on another non-key column "
        "(no transitive dependencies).", bullet_style)),
], bulletType="bullet", start="circle", leftIndent=14))
story.append(Paragraph(
    "Example: storing both a customer's city and that city's country in an orders "
    "table violates 3NF, because country depends on city, not on the order itself.",
    body,
))

story.append(PageBreak())

# ---------------- Page 5: CAP theorem ----------------
story.append(Paragraph("The CAP Theorem", h2))
story.append(Paragraph(
    "The CAP theorem states that a distributed data store can provide at most two "
    "of the following three guarantees at the same time, during a network partition:",
    body,
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Consistency</b> - every read receives the most recent write or an error.",
        bullet_style)),
    ListItem(Paragraph(
        "<b>Availability</b> - every request receives a non-error response, without "
        "the guarantee that it contains the most recent write.", bullet_style)),
    ListItem(Paragraph(
        "<b>Partition tolerance</b> - the system continues to operate despite an "
        "arbitrary number of messages being dropped or delayed between nodes.",
        bullet_style)),
], bulletType="bullet", start="circle", leftIndent=14))
story.append(Paragraph(
    "Because real networks do experience partitions, the practical choice is between "
    "CP (consistent but may reject requests during a partition) and AP (available but "
    "may serve stale data during a partition). A single-node relational database "
    "sidesteps this trade-off entirely, since there is no partition to tolerate.",
    body,
))

doc.build(story)
print("wrote", OUT)
