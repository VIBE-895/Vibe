# Created by guxu at 3/27/25
from enum import Enum


class NoteType(Enum):
    MEETING = 1
    CLASS = 2
    DIALOGUE = 3
    UNDEFINED = 99


class SupportingDocumentType(Enum):
    AUDIO = 1
    IMAGE = 2
    PDF = 3
    TEXT = 4
    VIDEO = 5
    UNKNOWN = 99