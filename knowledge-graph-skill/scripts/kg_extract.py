"""
KG Skill Extraction Module - Document -> Knowledge Graph pipeline.

Implements hybrid extraction (rule-based + LLM-assisted):
- Document parsing and chunking (text, markdown, JSON, table)
- Rule-based NER (regex patterns for common entity types)
- LLM-assisted extraction (OpenAI API for complex relations)
- Triple generation with confidence scoring
- Source Grounding (binding triples to source chunks)
"""

import json
import re
import os
from typing import Optional

from kg_core import Entity, Relation, Document, Chunk, KGStore, _generate_id, _now_iso
from kg_pii import PIIDetector


# ---------------------------------------------------------------------------
# Document Parser
# ---------------------------------------------------------------------------

class DocumentParser:
    """Parse documents into chunks based on format."""

    def parse(self, content: str, fmt: str = "auto",
              chunk_size: int = 512, overlap: int = 64) -> list:
        """Parse content into chunks. Returns list of (text, index)."""
        if fmt == "auto":
            fmt = self._detect_format(content)

        if fmt == "markdown":
            return self._parse_markdown(content, chunk_size, overlap)
        elif fmt == "json":
            return self._parse_json(content)
        elif fmt == "table":
            return self._parse_table(content)
        else:
            return self._parse_text(content, chunk_size, overlap)

    def _detect_format(self, content: str) -> str:
        """Auto-detect document format from content."""
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        if "|" in content and "---" in content:
            return "table"
        if content.count("#") > 2 and ("\n##" in content or "\n#" in content):
            return "markdown"
        return "text"

    def _parse_text(self, content: str, chunk_size: int,
                    overlap: int) -> list:
        """Split plain text into overlapping chunks by character count."""
        chunks = []
        start = 0
        idx = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, idx))
                idx += 1
            start = end - overlap if end < len(content) else end
        return chunks

    def _parse_markdown(self, content: str, chunk_size: int,
                        overlap: int) -> list:
        """Split markdown by headings, preserving structure."""
        sections = re.split(r'(?=^#{1,6}\s)', content, flags=re.MULTILINE)
        chunks = []
        idx = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) > chunk_size * 2:
                sub_chunks = self._parse_text(section, chunk_size, overlap)
                for text, _ in sub_chunks:
                    chunks.append((text, idx))
                    idx += 1
            else:
                chunks.append((section, idx))
                idx += 1
        return chunks

    def _parse_json(self, content: str) -> list:
        """Parse JSON content. Each top-level item becomes a chunk."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._parse_text(content, 512, 64)

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = [data]

        chunks = []
        for idx, item in enumerate(data):
            chunks.append((json.dumps(item, ensure_ascii=False, indent=2), idx))
        return chunks

    def _parse_table(self, content: str) -> list:
        """Parse markdown table format. Each row becomes a chunk."""
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return [(content, 0)]
        # First line is header, second is separator
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        chunks = [(content, 0)]  # Keep full table as first chunk
        for idx, line in enumerate(lines[2:], 1):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                row_dict = dict(zip(headers, cells))
                chunks.append((json.dumps(row_dict, ensure_ascii=False), idx))
        return chunks


# ---------------------------------------------------------------------------
# Rule-Based Extractor
# ---------------------------------------------------------------------------

class RuleExtractor:
    """Fast, low-cost entity and relation extraction using patterns."""

    # Common entity patterns (extensible)
    PATTERNS = {
        "email": (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), "Concept"),
        "url": (re.compile(r'https?://[\w./\-?=&]+'), "Concept"),
        "date": (re.compile(
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/\d{4})\b'), "Other"),
        "money": (re.compile(
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?'), "Other"),
        "phone": (re.compile(r'\+?\d[\d\s\-()]{8,}\d'), "Other"),
        # 中文日期
        "cn_date": (re.compile(
            r'(\d{4})\s*(?:年|-|/)\s*(\d{1,2})\s*(?:月|-|/)\s*(\d{1,2})\s*日?'), "Other"),
        # 中文金额
        "cn_money": (re.compile(
            r'(?:人民币|约|近|超)?\s*[\d,]+(?:\.\d+)?\s*(?:亿|万|千)元?'), "Other"),
    }

    # 中文组织名后缀 - 用于从文本中识别中文组织实体
    CN_ORG_SUFFIXES = [
        "公司", "集团", "银行", "大学", "学院", "研究院", "研究所",
        "医院", "基金会", "协会", "联盟", "中心", "实验室", "科技",
        "有限", "股份", "有限合伙",
    ]

    # Relation patterns (subject -> predicate -> object)
    RELATION_PATTERNS = [
        # English patterns
        (re.compile(
            r'(\w[\w\s]+?)\s+(?:was\s+)?founded\s+(?:by|in)\s+([\w\s,]+)',
            re.IGNORECASE), "FOUNDED_BY"),
        (re.compile(
            r'(\w[\w\s]+?)\s+(?:was\s+)?acquired\s+(?:by\s+)?([\w\s,]+)',
            re.IGNORECASE), "ACQUIRED_BY"),
        (re.compile(
            r'(\w[\w\s]+?)\s+(?:is\s+)?(?:CEO|CTO|CFO|founder)\s+of\s+([\w\s,]+)',
            re.IGNORECASE), "EXECUTIVE_OF"),
        # Chinese relation patterns
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*成立于\s*(\d{4})\s*年'), "FOUNDED_IN"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:由|被)\s*'
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:创立|创办|创建|成立)'),
            "FOUNDED_BY"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:收购|并购|兼并)\s*(?:了\s*)?'
            r'([\u4e00-\u9fff\w]{2,20}?)'), "ACQUIRED"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,10}?)\s*(?:担任|出任|任)\s*'
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:的)?\s*'
            r'(?:CEO|首席执行官|董事长|总裁|CTO|CFO|创始人|总经理)'),
            "EXECUTIVE_OF"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:总部|位于|坐落于)\s*'
            r'([\u4e00-\u9fff\w]{2,10}?)'), "LOCATED_IN"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:投资|注资)\s*(?:了\s*)?'
            r'([\u4e00-\u9fff\w]{2,20}?)'), "INVESTED_IN"),
        (re.compile(
            r'([\u4e00-\u9fff\w]{2,20}?)\s*(?:生产|研发|推出|发布)\s*(?:了\s*)?'
            r'([\u4e00-\u9fff\w\(\)0-9A-Za-z]{2,30}?)'), "PRODUCES"),
    ]

    def extract(self, text: str) -> dict:
        """Extract entities and relations using regex patterns."""
        entities = []
        relations = []

        # Entity extraction via patterns
        for ent_type, (pattern, default_type) in self.PATTERNS.items():
            for match in pattern.finditer(text):
                val = match.group(0)
                entities.append({
                    "name": val,
                    "type": default_type,
                    "confidence": 0.7,
                    "extraction_method": "rule-based",
                })

        # Capitalized word sequences as potential named entities (English)
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in cap_pattern.finditer(text):
            name = match.group(1)
            if name not in [e["name"] for e in entities]:
                entities.append({
                    "name": name,
                    "type": "Other",
                    "confidence": 0.5,
                    "extraction_method": "rule-based",
                })

        # 中文组织实体抽取（通过后缀匹配）
        existing_names = {e["name"] for e in entities}
        org_suffix_alt = "|".join(self.CN_ORG_SUFFIXES)
        cn_org_pattern = re.compile(
            r'([\u4e00-\u9fff]{2,15}(?:' + org_suffix_alt + r'))'
        )
        for match in cn_org_pattern.finditer(text):
            name = match.group(1)
            if name not in existing_names:
                entities.append({
                    "name": name,
                    "type": "Organization",
                    "confidence": 0.6,
                    "extraction_method": "rule-based",
                })
                existing_names.add(name)

        # 中文人名抽取（2-4 字汉字，出现在关系模式附近）
        # 简单启发式：匹配 "XX担任/出任/任" 或 "XX创立/创办" 前的人名
        cn_person_pattern = re.compile(
            r'([\u4e00-\u9fff]{2,4})\s*(?:担任|出任|任|创立|创办|创建|'
            r'投资|收购|领导|创建|发明)'
        )
        for match in cn_person_pattern.finditer(text):
            name = match.group(1)
            if name not in existing_names:
                entities.append({
                    "name": name,
                    "type": "Person",
                    "confidence": 0.5,
                    "extraction_method": "rule-based",
                })
                existing_names.add(name)

        # Relation extraction via patterns
        for pattern, rel_type in self.RELATION_PATTERNS:
            for match in pattern.finditer(text):
                subj = match.group(1).strip()
                obj = match.group(2).strip()
                # Clean trailing punctuation
                obj = re.sub(r'[.,;，。；、].*$', '', obj).strip()
                relations.append({
                    "subject_name": subj,
                    "relation_type": rel_type,
                    "object_name": obj,
                    "confidence": 0.7,
                    "extraction_method": "rule-based",
                })

        return {"entities": entities, "relations": relations}


# ---------------------------------------------------------------------------
# LLM-Assisted Extractor
# ---------------------------------------------------------------------------

class LLMExtractor:
    """High-accuracy extraction using LLM (OpenAI API)."""

    EXTRACTION_PROMPT = """Extract entities, relations, and events from the following text.

Return a JSON object with this structure:
{
  "entities": [
    {"name": "...", "type": "Organization|Person|Product|Event|Location|Concept|Other", "description": "...", "aliases": [], "attributes": {}}
  ],
  "relations": [
    {"subject_name": "...", "relation_type": "UPPER_SNAKE_CASE", "object_name": "...", "attributes": {}, "confidence": 0.0-1.0}
  ],
  "events": [
    {"event_type": "Acquisition|Merger|Launch|Meeting|Transaction|Other", "participants": [{"name": "...", "role": "..."}], "attributes": {}, "occurred_at": "YYYY-MM-DD"}
  ]
}

Rules:
- relation_type must be UPPER_SNAKE_CASE (e.g., ACQUIRED, WORKS_AT, PRODUCES)
- Include confidence scores (0-1) for each relation
- Extract aliases and attributes when available
- Only extract explicitly stated facts, do not infer

Text:
{text}"""

    def __init__(self, model: str = "gpt-4o", api_key: str = None,
                 api_base: str = None):
        """Initialize LLM extractor with OpenAI client."""
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                self.client = OpenAI(**kwargs)
            except ImportError:
                pass

    def extract(self, text: str) -> dict:
        """Extract entities and relations using LLM."""
        if not self.client:
            return {"entities": [], "relations": [], "events": []}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph "
                     "extraction assistant. Return valid JSON only."},
                    {"role": "user", "content": self.EXTRACTION_PROMPT.format(
                        text=text)},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            # Tag extraction method
            for e in result.get("entities", []):
                e["extraction_method"] = "llm-assisted"
                e.setdefault("confidence", 0.85)
            for r in result.get("relations", []):
                r["extraction_method"] = "llm-assisted"
                r.setdefault("confidence", 0.8)
            return result
        except Exception as e:
            return {"entities": [], "relations": [], "events": [],
                    "error": str(e)}


# ---------------------------------------------------------------------------
# Extraction Pipeline
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """Orchestrates hybrid extraction: rules first, LLM second."""

    def __init__(self, store: KGStore, config: dict = None):
        """Initialize pipeline with store and configuration."""
        self.store = store
        self.config = config or {}
        self.parser = DocumentParser()
        self.rule_extractor = RuleExtractor()
        self.llm_extractor = LLMExtractor(
            model=self.config.get("llm", {}).get("model", "gpt-4o"),
            api_base=self.config.get("llm", {}).get("api_base", ""),
        )
        self.pii_detector = PIIDetector() if self.config.get(
            "security", {}).get("pii_detection", True) else None

    def extract(self, content: str, fmt: str = "auto",
                strategy: str = "auto",
                chunk_size: int = 512, chunk_overlap: int = 64,
                auto_resolve: bool = True,
                title: str = "") -> dict:
        """
        Full extraction pipeline: parse -> chunk -> extract -> resolve -> persist.

        Returns summary dict with counts and extracted items.
        """
        # Save document
        doc = Document(
            doc_id=_generate_id("doc"),
            title=title or "Untitled",
            format=fmt,
            content=content[:10000],  # Truncate for storage
        )
        self.store.save_document(doc)

        # PII detection and masking (before extraction)
        pii_report = []
        if self.pii_detector:
            content, pii_report = self.pii_detector.detect_and_mask(content)

        # Parse and chunk
        chunks = self.parser.parse(content, fmt, chunk_size, chunk_overlap)

        all_entities = {}
        all_relations = []
        all_events = []
        llm_tokens = 0

        for chunk_text, chunk_idx in chunks:
            # Save chunk
            chunk = Chunk(
                chunk_id=_generate_id("chunk"),
                doc_id=doc.doc_id,
                text=chunk_text,
                chunk_index=chunk_idx,
            )
            self.store.save_chunk(chunk)

            # Rule-based extraction (always run, cheap)
            rule_result = self.rule_extractor.extract(chunk_text)

            # Decide whether to also run LLM
            use_llm = strategy == "llm_first" or (
                strategy == "auto" and (
                    len(rule_result["entities"]) < 2 or
                    len(rule_result["relations"]) == 0
                )
            )

            llm_result = {"entities": [], "relations": [], "events": []}
            if use_llm and self.llm_extractor.client:
                llm_result = self.llm_extractor.extract(chunk_text)

            # Merge results (LLM takes priority for duplicates)
            for ent_data in rule_result["entities"] + llm_result["entities"]:
                name = ent_data["name"]
                if name not in all_entities:
                    all_entities[name] = {**ent_data,
                                          "source_chunk_id": chunk.chunk_id}
                elif ent_data.get("extraction_method") == "llm-assisted":
                    all_entities[name] = {**ent_data,
                                          "source_chunk_id": chunk.chunk_id}

            for rel_data in rule_result["relations"] + llm_result["relations"]:
                rel_data["source_chunk_id"] = chunk.chunk_id
                all_relations.append(rel_data)

            for evt_data in llm_result.get("events", []):
                evt_data["source_chunk_id"] = chunk.chunk_id
                all_events.append(evt_data)

        # Persist entities
        entity_name_to_id = {}
        created_entities = []
        for name, ent_data in all_entities.items():
            entity = Entity(
                name=ent_data["name"],
                type=ent_data.get("type", "Other"),
                aliases=ent_data.get("aliases", []),
                attributes=ent_data.get("attributes", {}),
                description=ent_data.get("description", ""),
                confidence=ent_data.get("confidence", 0.7),
                provenance={
                    "source_doc_id": doc.doc_id,
                    "source_chunk_id": ent_data.get("source_chunk_id"),
                    "extraction_method": ent_data.get("extraction_method",
                                                       "rule-based"),
                    "extracted_at": _now_iso(),
                },
            )
            result = self.store.create_entity(entity)
            entity_name_to_id[name] = result["entity_id"]
            created_entities.append({
                "entity_id": result["entity_id"],
                "name": ent_data["name"],
                "type": ent_data.get("type", "Other"),
                "confidence": ent_data.get("confidence", 0.7),
                "status": result["status"],
            })

        # Persist relations
        created_relations = []
        for rel_data in all_relations:
            src_name = rel_data.get("subject_name", "")
            tgt_name = rel_data.get("object_name", "")
            src_id = entity_name_to_id.get(src_name)
            tgt_id = entity_name_to_id.get(tgt_name)

            if not src_id or not tgt_id:
                continue

            relation = Relation(
                source_entity_id=src_id,
                relation_type=rel_data.get("relation_type", "RELATED_TO"),
                target_entity_id=tgt_id,
                attributes=rel_data.get("attributes", {}),
                confidence=rel_data.get("confidence", 0.7),
                provenance={
                    "source_doc_id": doc.doc_id,
                    "source_chunk_id": rel_data.get("source_chunk_id"),
                    "extraction_method": rel_data.get("extraction_method",
                                                       "rule-based"),
                    "extracted_at": _now_iso(),
                },
            )
            result = self.store.create_relation(relation)
            if result.get("status") == "created":
                created_relations.append({
                    "relation_id": result["relation_id"],
                    "type": rel_data.get("relation_type"),
                    "confidence": rel_data.get("confidence", 0.7),
                })

        merged_count = sum(
            1 for e in created_entities if e["status"] == "merged"
        )

        return {
            "doc_id": doc.doc_id,
            "status": "completed",
            "pii_masked": pii_report if pii_report else None,
            "summary": {
                "chunks_created": len(chunks),
                "entities_extracted": len(created_entities),
                "relations_extracted": len(created_relations),
                "events_extracted": len(all_events),
                "entities_merged": merged_count,
                "avg_confidence": round(
                    sum(e["confidence"] for e in created_entities) /
                    max(len(created_entities), 1), 4
                ),
            },
            "entities": created_entities,
            "relations": created_relations,
        }
