"""The interview schedule as data. INTERVIEW.md §4 — four blocks, their
opening lines, and every required coverage point (A1-A4, B1-B8, C1-C3,
D1-D2) with the open-ended question that opens it.

This is a *schedule*, not a verbatim question sheet (INTERVIEW.md §2:
"訪談大綱...不是逐字題本") — the interviewer (`ingest.interviewer`) follows up
on each point until it reaches a concrete instance; the wording here is only
the first question asked. Questions are open-ended and offer no options
(§6.1 "MUST NOT 誘導...MUST NOT 提供選項").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoveragePoint:
    """One INTERVIEW.md §4 required coverage point. `required_instances` is
    how many concrete instances (§2: 有時間、有地點、有結果的真實事件) the
    point needs before it counts as reached — 3 for B1/B2/B6 (§7 Q2), 1 for
    points that ask for "一個事例", 0 for pure self-description points."""

    point_id: str
    block: str
    question: str
    required_instances: int


BLOCK_OPENINGS: dict[str, str] = {
    "A": "請跟我說說你的人生故事，從哪裡開始都可以。",  # §4 A: opening MUST be this open-ended line
    "B": "接下來我想聊聊你平常怎麼決定「回不回」「查不查」「發不發」——都用實際發生過的事來說。",
    "C": "接下來請你回想幾件事。不用管日期精確不精確，照你記得的方式說就好。",
    "D": "最後幾個問題。",
}

COVERAGE_POINTS: tuple[CoveragePoint, ...] = (
    CoveragePoint("A1", "A", "到目前為止，你覺得人生裡有哪幾個轉折點？每一個大概是什麼時候、當時有哪些選項、你為什麼那樣選？（至少三個）", 3),
    CoveragePoint("A2", "A", "有沒有哪些決定，後來回頭看覺得當初選錯了？說兩件，發生了什麼事。", 2),
    CoveragePoint("A3", "A", "你覺得自己跟身邊的人最不一樣的地方是什麼？", 0),
    CoveragePoint("A4", "A", "有哪兩件事你明確很在乎、哪兩件事你明確不在乎？每一項舉一個實際發生過的例子。", 4),
    CoveragePoint("B1", "B", "說三次你本來想回訊息、最後沒回的事。當時是什麼情況、為什麼沒回？", 3),
    CoveragePoint("B2", "B", "說三次你其實不必回、但還是回了的事。當時為什麼回？", 3),
    CoveragePoint("B3", "B", "遇到不確定的事，你什麼時候會去查、什麼時候不查？各說一個真的發生過的例子。", 2),
    CoveragePoint("B4", "B", "查了但查不到的時候，你通常試到第幾次就放棄？說一次你放棄的經驗。", 1),
    CoveragePoint("B5", "B", "什麼樣的內容會讓你主動想發言或發文？最近一次是什麼時候、發了什麼？", 1),
    CoveragePoint("B6", "B", "什麼樣的內容你有感觸、但最後決定不發？說三次，每次當下在想什麼。", 3),
    CoveragePoint("B7", "B", "你會主動推薦別人用什麼東西？通常怎麼推？舉一次實際推薦的經過。", 1),
    CoveragePoint("B8", "B", "對家人、同事、陌生人，你回訊息的速度跟語氣有什麼不一樣？各舉一個例子。", 3),
    CoveragePoint("C1", "C", "請從三個不同的人生時期各回想一件事，照你記得的樣子說。", 3),
    CoveragePoint("C2", "C", "剛才說的這幾件事，你會怎麼形容它們大概是什麼時候發生的？", 0),
    CoveragePoint("C3", "C", "剛才這些事裡，有哪個部分你其實不太確定、可能記錯了？", 0),
    CoveragePoint("D1", "D", "有什麼是我沒問到、但你覺得要理解你就必須知道的？", 0),
    CoveragePoint("D2", "D", "如果有個東西要代替你回訊息，你最不希望它做什麼？", 0),
)

BLOCK_ORDER: tuple[str, ...] = ("A", "B", "C", "D")


def points_in_block(block: str) -> list[CoveragePoint]:
    return [p for p in COVERAGE_POINTS if p.block == block]
