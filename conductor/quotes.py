"""Whiplash-flavored one-liners shown in the HUD on events.

Short single-sentence quotations from the film, used as easter-egg flavor
text. Buckets are keyed by event type; pick() returns one at random.
"""
import random

HALT = [
    "Why do you suppose I just hurled a chair at you, Niemann?",
    "Not my tempo.",
    "Were you rushing or were you dragging?",
]

OFF_TEMPO = [
    "Not quite my tempo.",
    "Were you rushing or were you dragging?",
]

START = [
    "Alright. Let's go.",
    "From the top.",
]

ON_TEMPO = [
    "Yes. There it is.",
]


def pick(bucket):
    return random.choice(bucket) if bucket else ""
