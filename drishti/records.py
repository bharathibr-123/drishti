"""Shared record types. Everything stored on disk or sent to the browser has a shape here."""

from __future__ import annotations

from typing import Literal, TypedDict

Box = list[int]  # [x1, y1, x2, y2] in source-frame pixels
Label = Literal["strong", "possible", "weak"]


class Camera(TypedDict, total=False):
    id: str
    name: str
    video: str
    start_time: str  # ISO wall-clock time when the recording starts
    x: float  # position on the site map, 0-100
    y: float  # 0-80
    live: bool  # streamed for live alerts instead of indexed
    duration: float


class Track(TypedDict):
    key: str  # "<camera>:<track id>"
    camera_id: str
    t_start: float
    t_end: float
    hits: int
    thumb: str
    path: list[list[float]]  # [t, centre x, centre y] samples, x/y as fractions of the frame


class Crop(TypedDict):
    track: str
    camera_id: str
    t: float
    box: Box
    frame_size: list[int]
    thumb: str
    head: list[float]  # this person's head box as fractions of the crop (privacy blur)
    heads: list[list[float]]  # head boxes of everyone in that frame, frame pixels (privacy blur)


class Hit(TypedDict):
    track: str
    camera_id: str
    camera: str
    score: float
    z: float
    label: Label
    same_look: float  # appearance similarity to the #1 hit
    on_trail: bool
    thumb: str
    t: float
    t_start: float
    t_end: float
    clock: str | None
    clock_start: str | None
    clock_end: str | None


class TrailStop(TypedDict):
    camera_id: str
    camera: str
    clock_start: str
    clock_end: str
    thumb: str
    track: str


class SearchResult(TypedDict):
    query: str
    understood_as: str
    language: str
    attributes: list[str]
    results: list[Hit]
    trail: list[TrailStop]
    last_seen: TrailStop | None
