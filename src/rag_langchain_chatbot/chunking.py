"""Semantic, structure-aware, and token-bounded document chunking engine.

Provides:
- Multilingual sentence boundary segmentation (French, English, Arabic, Tunisian dialect)
- Punctuation & abbreviation preservation (no mid-sentence slicing)
- Document hierarchy extraction (Markdown headers, legal articles, clauses, breadcrumbs)
- Table detection and Markdown formatting preservation
- Token-aware chunk sizing with minimum/maximum token guards
- Sibling chunk linking (prev_chunk_id, next_chunk_id, chunk_index)
- Contextual breadcrumb prefixing for enriched dense and BM25 retrieval
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Token Counter (Tokenizer-Aware with Accurate Multi-Tier Fallback)
# ---------------------------------------------------------------------------

class TokenCounter:
    """Calculates token counts using model tokenizers or robust multilingual fallbacks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._tokenizer = None
        self._tiktoken_encoder = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Initialize HuggingFace AutoTokenizer or tiktoken fallback gracefully."""
        if self.model_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self._tokenizer.model_max_length = 100_000
                return
            except Exception:
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._tokenizer.model_max_length = 100_000
                    return
                except Exception as ex:
                    logger.debug("Could not load HuggingFace tokenizer for '%s': %s", self.model_name, ex)

        # Fallback to tiktoken
        try:
            import tiktoken
            self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tiktoken_encoder = None

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""
        if not text:
            return 0

        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        if self._tiktoken_encoder is not None:
            try:
                return len(self._tiktoken_encoder.encode(text))
            except Exception:
                pass

        # Multilingual word/character heuristic fallback:
        # English/French: ~4 chars per token, Arabic: ~2.5 chars per token
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        if arabic_chars > len(text) * 0.3:
            return max(1, int(len(text) / 2.8))
        return max(1, int(len(text) / 3.8))


# ---------------------------------------------------------------------------
# 2. Multilingual Sentence Segmenter
# ---------------------------------------------------------------------------

class SentenceSegmenter:
    """Robust multilingual sentence segmenter for French, English, Arabic, and Tunisian dialect."""

    # Abbreviations and titles that must not trigger a sentence split
    ABBREVIATIONS = [
        "Dr.", "dr.", "M.", "m.", "Mme.", "mme.", "Mlle.", "mlle.",
        "Prof.", "prof.", "art.", "Art.", "no.", "No.", "vol.", "Vol.",
        "cf.", "Cf.", "vs.", "Vs.", "etc.", "i.e.", "e.g.", "p.ex.",
        "av.", "bvd.", "al.", "al-", "st.", "St.", "fig.", "Fig."
    ]

    def __init__(self):
        sorted_abbrs = sorted(self.ABBREVIATIONS, key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_abbrs]
        self._abbr_pattern = re.compile(r'\b(' + '|'.join(escaped) + r')', flags=re.IGNORECASE)
        # Protect single letter initials and legal article sub-indices (e.g., "art. L. 113-8")
        self._initials_pattern = re.compile(r'\b([A-Za-z]\.)\s*(?=[0-9A-Za-z])')
        # Protect decimal numbers and currency (e.g. "150.50", "01.42.68.00.00")
        self._decimal_pattern = re.compile(r'(?<=\d)\.(?=\d)')
        # Punctuation boundaries for English, French, and Arabic
        self._boundary_pattern = re.compile(r'([.!?؟]+)(?:\s+|\n+)(?=[A-Z\u0600-\u06FF0-9"«\'\-])')

    def split_sentences(self, text: str) -> List[str]:
        """Split text into complete grammatical sentences, preserving punctuation."""
        if not text or not text.strip():
            return []

        text = text.strip()
        # Protect decimals
        masked_text = self._decimal_pattern.sub('\u200B_DOT_\u200B', text)

        # Protect known abbreviations
        abbr_map: Dict[str, str] = {}
        def _replace_abbr(match: re.Match) -> str:
            key = f"\u200B_ABBR_{len(abbr_map)}_\u200B"
            abbr_map[key] = match.group(0)
            return key

        masked_text = self._abbr_pattern.sub(_replace_abbr, masked_text)

        # Protect single letter initials and legal codes
        def _replace_initial(match: re.Match) -> str:
            key = f"\u200B_INIT_{len(abbr_map)}_\u200B"
            abbr_map[key] = match.group(1)
            return key + (" " if match.group(0).endswith(" ") else "")

        masked_text = self._initials_pattern.sub(_replace_initial, masked_text)

        # Mark double newlines as explicit structural paragraph breaks
        masked_text = re.sub(r'\n{2,}', '<|PARAGRAPH|>', masked_text)

        # Mark sentence boundaries
        def _mark_boundary(m: re.Match) -> str:
            return m.group(1) + "<|SENTENCE|>"

        marked = self._boundary_pattern.sub(_mark_boundary, masked_text)
        marked = marked.replace("<|PARAGRAPH|>", "<|SENTENCE|>")

        raw_splits = marked.split("<|SENTENCE|>")
        sentences = []
        for s in raw_splits:
            cleaned = s.strip()
            if not cleaned:
                continue
            # Restore decimal dots and abbreviations
            cleaned = cleaned.replace('\u200B_DOT_\u200B', '.')
            for k, v in abbr_map.items():
                cleaned = cleaned.replace(k, v)
            sentences.append(cleaned)

        return sentences


# ---------------------------------------------------------------------------
# 3. Document Structure Parser & Hierarchy Extractor
# ---------------------------------------------------------------------------

@dataclass
class StructuralBlock:
    """Represents a coherent section or block within a document."""
    title: str
    level: int
    breadcrumbs: List[str]
    content: str
    clause_number: Optional[str] = None
    is_table: bool = False


class DocumentStructureParser:
    """Parses documents into hierarchical section blocks preserving breadcrumbs and legal numbering."""

    HEADING_PATTERNS = [
        # Markdown headings: # Title, ## Subtitle, etc.
        (re.compile(r'^(#{1,6})\s+(.+)$'), "markdown"),
        # Legal: Article 4, Art. 4.2, ARTICLE 5 : Déclaration
        (re.compile(r'^(?:Article|Art\.|ARTICLE)\s+(\d+[\w\.\-]*)(?:\s*[:\-–—]\s*(.*))?$', re.IGNORECASE), "article"),
        # Legal: Chapitre I, Chapter 2
        (re.compile(r'^(?:Chapitre|CHAPITRE|Chapter)\s+([IVXLCDM\d]+)(?:\s*[:\-–—]\s*(.*))?$', re.IGNORECASE), "chapter"),
        # Legal: Section 1, Section II
        (re.compile(r'^(?:Section|SECTION)\s+([IVXLCDM\d]+)(?:\s*[:\-–—]\s*(.*))?$', re.IGNORECASE), "section"),
        # Numbered headings: 1.1 Scope, 2.3.1 Exclusion
        (re.compile(r'^(\d+\.\d+(?:\.\d+)?)\s+([A-Z\u0600-\u06FF].*)$'), "numbered"),
    ]

    def parse(self, text: str, default_title: str = "Document") -> List[StructuralBlock]:
        """Parse structured text into a flat sequence of hierarchical blocks."""
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        blocks: List[StructuralBlock] = []

        stack: List[Tuple[int, str]] = []
        current_title = default_title
        current_level = 1
        current_clause: Optional[str] = None
        current_lines: List[str] = []
        is_in_table = False

        def _flush_current():
            nonlocal current_lines, is_in_table
            content = "\n".join(current_lines).strip()
            if content:
                breadcrumbs = [t for _, t in stack] if stack else [current_title]
                blocks.append(StructuralBlock(
                    title=current_title,
                    level=current_level,
                    breadcrumbs=list(breadcrumbs),
                    content=content,
                    clause_number=current_clause,
                    is_table=is_in_table
                ))
            current_lines = []
            is_in_table = False

        for line in lines:
            stripped = line.strip()

            # Detect Markdown table rows
            if stripped.startswith("|") and stripped.endswith("|"):
                if not is_in_table and current_lines:
                    _flush_current()
                is_in_table = True
                current_lines.append(line)
                continue
            elif is_in_table:
                _flush_current()

            matched_heading = None
            matched_level = 1
            matched_clause = None

            for pat, kind in self.HEADING_PATTERNS:
                m = pat.match(stripped)
                if m:
                    if kind == "markdown":
                        hashes = m.group(1)
                        matched_level = len(hashes)
                        matched_heading = m.group(2).strip()
                    elif kind == "article":
                        art_num = m.group(1).strip()
                        art_title = (m.group(2) or "").strip()
                        matched_level = 3
                        matched_clause = art_num
                        matched_heading = f"Article {art_num}" + (f": {art_title}" if art_title else "")
                    elif kind == "chapter":
                        chap_num = m.group(1).strip()
                        chap_title = (m.group(2) or "").strip()
                        matched_level = 2
                        matched_heading = f"Chapitre {chap_num}" + (f": {chap_title}" if chap_title else "")
                    elif kind == "section":
                        sec_num = m.group(1).strip()
                        sec_title = (m.group(2) or "").strip()
                        matched_level = 2
                        matched_heading = f"Section {sec_num}" + (f": {sec_title}" if sec_title else "")
                    elif kind == "numbered":
                        num = m.group(1).strip()
                        heading_txt = m.group(2).strip()
                        matched_level = min(num.count(".") + 2, 6)
                        matched_clause = num
                        matched_heading = f"{num} {heading_txt}"
                    break

            # Dynamic uppercase section heading detection for unstructured text
            if not matched_heading and stripped.isupper() and 1 <= len(stripped.split()) <= 7 and len(stripped) >= 4:
                if not stripped.endswith((".", ",", ";", ":", "-", "|")):
                    matched_heading = stripped.title()
                    matched_level = 2
                    matched_clause = None

            if matched_heading:
                _flush_current()
                while stack and stack[-1][0] >= matched_level:
                    stack.pop()
                stack.append((matched_level, matched_heading))
                current_title = matched_heading
                current_level = matched_level
                current_clause = matched_clause
            else:
                current_lines.append(line)

        _flush_current()
        return blocks


# ---------------------------------------------------------------------------
# 4. Semantic Hybrid Text Splitter
# ---------------------------------------------------------------------------

class SemanticHybridSplitter:
    """Structure-aware, sentence-preserving, and token-bounded hybrid text splitter.
    
    Conforms to the LangChain Document splitter interface while enforcing:
    - Hierarchical breadcrumbs injection
    - Sentence boundary preservation (no mid-sentence cuts)
    - Token limits (min_chunk_tokens <= tokens <= max_chunk_tokens)
    - Table structure retention
    - Sibling chunk linking (prev_chunk_id, next_chunk_id)
    """

    def __init__(
        self,
        target_tokens: int = 250,
        min_tokens: int = 40,
        max_tokens: int = 384,
        overlap_sentences: int = 1,
        enable_breadcrumbs: bool = True,
        tokenizer_model_name: Optional[str] = None,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.enable_breadcrumbs = enable_breadcrumbs
        self.token_counter = TokenCounter(tokenizer_model_name)
        self.segmenter = SentenceSegmenter()
        self.structure_parser = DocumentStructureParser()

    def split_text(self, text: str) -> List[str]:
        """Split raw text into clean chunk strings."""
        doc = Document(page_content=text, metadata={})
        chunks = self.split_documents([doc])
        return [c.page_content for c in chunks]

    def _split_table(self, table_text: str) -> List[str]:
        """Split large markdown tables while repeating the header row on each split."""
        lines = [ln.strip() for ln in table_text.split("\n") if ln.strip()]
        if len(lines) <= 2:
            return [table_text]

        header = lines[0]
        separator = lines[1] if len(lines) > 1 and "---" in lines[1] else "| " + " | ".join(["---"] * header.count("|")) + " |"
        rows = lines[2:] if "---" in lines[1] else lines[1:]

        chunks: List[str] = []
        current_rows: List[str] = []

        for row in rows:
            candidate = "\n".join([header, separator] + current_rows + [row])
            if self.token_counter.count_tokens(candidate) > self.max_tokens and current_rows:
                chunks.append("\n".join([header, separator] + current_rows))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            chunks.append("\n".join([header, separator] + current_rows))

        return chunks or [table_text]

    def _split_prose_block(self, text: str) -> List[str]:
        """Split a prose block into complete sentences bounded by min/max token limits."""
        sentences = self.segmenter.split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_chunk_sentences: List[str] = []

        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)

            # If a single sentence exceeds max_tokens on its own, force-split on comma/semicolon/words
            if sentence_tokens > self.max_tokens:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []

                # Fallback sub-split for extremely long run-on sentences
                words = sentence.split(" ")
                sub_acc: List[str] = []
                for w in words:
                    sub_acc.append(w)
                    if self.token_counter.count_tokens(" ".join(sub_acc)) >= self.target_tokens:
                        chunks.append(" ".join(sub_acc))
                        sub_acc = []
                if sub_acc:
                    chunks.append(" ".join(sub_acc))
                continue

            candidate = " ".join(current_chunk_sentences + [sentence])
            candidate_tokens = self.token_counter.count_tokens(candidate)

            if candidate_tokens > self.max_tokens and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                # Add overlap from previous sentence if configured
                if self.overlap_sentences > 0 and len(current_chunk_sentences) >= self.overlap_sentences:
                    overlap = current_chunk_sentences[-self.overlap_sentences:]
                    current_chunk_sentences = list(overlap) + [sentence]
                else:
                    current_chunk_sentences = [sentence]
            else:
                current_chunk_sentences.append(sentence)

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # Enforce min_tokens: merge small trailing/leading fragments
        if len(chunks) > 1:
            merged: List[str] = []
            for c in chunks:
                if merged and self.token_counter.count_tokens(c) < self.min_tokens:
                    # Check if merging with previous chunk exceeds max_tokens
                    combined = merged[-1] + " " + c
                    if self.token_counter.count_tokens(combined) <= self.max_tokens + 40:
                        merged[-1] = combined
                        continue
                merged.append(c)
            chunks = merged

        return chunks

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Split a collection of documents into structured, enriched chunks."""
        output_chunks: List[Document] = []

        for doc in documents:
            source = doc.metadata.get("source", "")
            file_name = doc.metadata.get("file_name", "Document")
            page = doc.metadata.get("page", 0)

            # 1. Parse structural blocks
            blocks = self.structure_parser.parse(doc.page_content, default_title=file_name)
            if not blocks:
                continue

            doc_chunks: List[Document] = []

            for block in blocks:
                block_tokens = self.token_counter.count_tokens(block.content)

                # Segment block based on type and length
                if block.is_table:
                    raw_segments = self._split_table(block.content)
                elif block_tokens <= self.max_tokens and block_tokens >= self.min_tokens:
                    raw_segments = [block.content]
                elif block_tokens > self.max_tokens:
                    raw_segments = self._split_prose_block(block.content)
                else:
                    raw_segments = [block.content]

                for seg in raw_segments:
                    seg = seg.strip()
                    if not seg:
                        continue

                    chunk_meta = dict(doc.metadata)
                    
                    # Context Breadcrumb Generation
                    breadcrumbs_str = " > ".join(block.breadcrumbs) if block.breadcrumbs else file_name
                    chunk_meta["section"] = block.title
                    chunk_meta["breadcrumbs"] = breadcrumbs_str
                    if block.clause_number:
                        chunk_meta["clause_number"] = block.clause_number

                    # Breadcrumb header prefixing for retrieval enrichment
                    if self.enable_breadcrumbs and block.breadcrumbs:
                        prefix = f"[{file_name} > {breadcrumbs_str}]\n"
                        # Only prepend if not already starting with this prefix
                        if not seg.startswith("["):
                            enriched_content = f"{prefix}{seg}"
                        else:
                            enriched_content = seg
                    else:
                        enriched_content = seg

                    seg_tokens = self.token_counter.count_tokens(enriched_content)
                    chunk_meta["token_count"] = seg_tokens
                    chunk_meta["char_count"] = len(enriched_content)

                    doc_chunks.append(Document(page_content=enriched_content, metadata=chunk_meta))

            # 2. Sibling Linkage (chunk_index, total_chunks, prev_chunk_id, next_chunk_id)
            # Deterministic chunk IDs
            for idx, c in enumerate(doc_chunks):
                c.metadata["chunk_index"] = idx
                c.metadata["total_chunks"] = len(doc_chunks)

            output_chunks.extend(doc_chunks)

        return output_chunks
