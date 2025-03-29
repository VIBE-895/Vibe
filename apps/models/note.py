from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from ..enums.enums import NoteType, SupportingDocumentType
from ..utils.utils import gen_id


@dataclass
class SupportingDocument:
    doc_type: SupportingDocumentType
    doc_url: str


@dataclass
class Note:
    id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    note_type: NoteType
    supporting_docs: List[SupportingDocument] = field(default_factory=list)


def create_note(data: dict) -> Note:
    docs_data = data.get("supporting_docs", [])
    docs = [
        SupportingDocument(
            doc_type=doc.get("doc_type"),
            doc_url=doc.get("doc_url")
        ) for doc in docs_data
    ]

    return Note(
        id=data.get("id", gen_id()),
        title=data.get("title", "untitled note"),
        content=data.get("content", ""),
        created_at=data.get("created_at", datetime.now()),
        updated_at=data.get("updated_at", datetime.now()),
        note_type=data.get("note_type", NoteType.UNDEFINED),
        supporting_docs=docs
    )
