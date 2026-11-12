"""The three synthetic evaluation conversations: generated once, committed, then only loaded.

Each conversation is a list of turns (LangChain message dicts) with planted facts, plus
questions about those facts with gold answers. The text is domain-neutral: personal projects
with invented names, no client, product or industry. The generator is deterministic (a fixed
seed), so `ccm-eval generate` rewrites the same files.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

type Turn = list[dict]

CONVERSATIONS = ("short", "long", "switching")


@dataclass(frozen=True)
class Question:
    question: str
    answer: str
    turn: int  # the turn (1-based) that plants the fact; asked only once it is replayed


@dataclass(frozen=True)
class Conversation:
    name: str
    description: str
    turns: list[Turn]
    questions: list[Question]


def load(directory: Path) -> list[Conversation]:
    conversations = []
    for name in CONVERSATIONS:
        data = json.loads((directory / f"{name}.json").read_text())
        questions = [Question(**q) for q in data.pop("questions")]
        conversations.append(Conversation(**data, questions=questions))
    return conversations


def write(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for conversation in generate():
        path = directory / f"{conversation.name}.json"
        path.write_text(json.dumps(asdict(conversation), indent=1, ensure_ascii=False) + "\n")


# A planted fact: the user's message, the assistant's answer, a question and its gold answer.
type Fact = tuple[str, str, str, str]

# Per topic: filler turns as (user, assistant) templates; {a} and {b} are filled from WORDS.
FILLER: dict[str, list[tuple[str, str]]] = {
    "garden": [
        (
            "Should I water the {a} in the morning or the evening?",
            "Morning is better for the {a}: the leaves dry before night, which keeps mould away.",
        ),
        (
            "The {a} look a bit pale. Is that a problem?",
            "Pale {a} often want more light or a little compost. Check the soil first.",
        ),
        (
            "How deep should the mulch be around the {a}?",
            "About five centimetres around the {a}, kept a hand's width from the stems.",
        ),
        (
            "Can the {a} grow next to the {b}?",
            "Yes, the {a} and the {b} get along; give each its own row so they share the light.",
        ),
        (
            "What do I do with the weeds between the {a}?",
            "Pull them by hand after rain, when the roots come out whole.",
        ),
        (
            "Is it too late to sow more {a}?",
            "Not yet. A second sowing of {a} still has time before the nights turn cold.",
        ),
    ],
    "archive": [
        (
            "Can you list the photos in the {a} folder again?",
            "Here is the {a} folder; most files are scans, a few are originals from the camera.",
        ),
        (
            "Which files in the {a} set are duplicates?",
            "The duplicates in the {a} set share a size and a date; I marked them for review.",
        ),
        (
            "Sort the {a} pictures by date, please.",
            "Done: the {a} pictures now run from the oldest scan to the newest.",
        ),
        (
            "Are there pictures of the {b} among the {a} scans?",
            "A handful of the {a} scans show the {b}; they are tagged now.",
        ),
        (
            "How large is the {a} folder in total?",
            "The {a} folder is a few gigabytes; the raw camera files take most of the space.",
        ),
    ],
    "trip": [
        (
            "What should I pack for the {a}?",
            "For the {a}: layers, a rain jacket and shoes you can walk in all day.",
        ),
        (
            "Is the {a} worth a whole day?",
            "The {a} fills half a day comfortably; pair it with the {b} for the afternoon.",
        ),
        (
            "How do we get from the station to the {a}?",
            "A local bus goes to the {a} every half hour; walking takes about forty minutes.",
        ),
        (
            "Any tips for eating near the {a}?",
            "Near the {a}, the small places on the side streets are quieter than the main square.",
        ),
    ],
    "piano": [
        (
            "My left hand is still slow in the {a}. Any advice?",
            "Practise the {a} with the left hand alone, slowly, then add the right hand.",
        ),
        (
            "How long should I spend on the {a} each day?",
            "Ten focused minutes on the {a} beat half an hour of running through it.",
        ),
        (
            "Should I use the metronome for the {a}?",
            "Yes, start the {a} well below tempo and raise it a few beats per week.",
        ),
        (
            "The {a} sounds uneven. Why?",
            "Uneven {a} usually means the thumb is late; play it in rhythms to even it out.",
        ),
    ],
    "paint": [
        (
            "Do I need to sand the {a} before painting?",
            "A light sanding of the {a} helps the paint grip; wipe the dust off afterwards.",
        ),
        (
            "How many coats does the {a} need?",
            "Two thin coats on the {a} look better than one thick one.",
        ),
        (
            "Can I paint the {a} when it is humid?",
            "Better wait: on a humid day the {a} dries slowly and can stay tacky.",
        ),
        (
            "What brush works best for the {a}?",
            "An angled brush for the edges of the {a}, a roller for the flat parts.",
        ),
    ],
}

WORDS: dict[str, list[str]] = {
    "garden": ["beans", "carrots", "lettuce", "tomatoes", "courgettes", "radishes", "onions"],
    "archive": ["summer", "school", "wedding", "harbour", "winter", "garden", "birthday"],
    "trip": ["old town", "lighthouse", "market", "cliff path", "museum", "botanical garden"],
    "piano": ["scales", "arpeggios", "waltz", "etude", "chord changes", "minuet"],
    "paint": ["skirting boards", "window frames", "ceiling", "door", "radiator", "wall"],
}


def _filler(rng: random.Random, topic: str) -> Turn:
    user, assistant = rng.choice(FILLER[topic])
    a, b = rng.sample(WORDS[topic], 2)
    return _turn(user.format(a=a, b=b), assistant.format(a=a, b=b))


def _turn(user: str, assistant: str) -> Turn:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _plant(turns: list[Turn], questions: list[Question], fact: Fact) -> None:
    user, assistant, question, answer = fact
    turns.append(_turn(user, assistant))
    questions.append(Question(question, answer, len(turns)))


def _answer(text: str) -> dict:
    return {"role": "assistant", "content": text}


GARDEN_FACTS: list[Fact] = [
    (
        "The garden plot is number 17 at the Linden Lane allotments.",
        "Noted: plot 17 at Linden Lane.",
        "Which plot number is the garden?",
        "Plot 17 at the Linden Lane allotments.",
    ),
    (
        "The raised bed I built is 240 cm long and 90 cm wide.",
        "A 240 by 90 cm bed gives you room for three short rows.",
        "How big is the raised bed?",
        "240 cm long and 90 cm wide.",
    ),
    (
        "My neighbour on the plot, Agnes Varro, lends me her wheelbarrow.",
        "Good to have Agnes Varro next door.",
        "Who is the neighbour that lends a wheelbarrow?",
        "Agnes Varro.",
    ),
    (
        "I sowed the first beans on 14 April.",
        "Beans sown on 14 April should flower by early June.",
        "When were the first beans sown?",
        "On 14 April.",
    ),
    (
        "The tomato variety I chose is called Copper Lantern.",
        "Copper Lantern it is; it likes a sheltered spot.",
        "Which tomato variety was chosen?",
        "Copper Lantern.",
    ),
    (
        "The water tap is at the north gate, about 35 metres from my bed.",
        "35 metres is a long walk with cans; a hose may pay off.",
        "Where is the water tap and how far from the bed?",
        "At the north gate, about 35 metres from the bed.",
    ),
    (
        "We decided to skip potatoes this year because of the blight last season.",
        "Skipping potatoes for a year is wise after blight.",
        "Why were potatoes skipped this year?",
        "Because of the blight last season.",
    ),
    (
        "The plot rent is 46 euros per year, due every March.",
        "So 46 euros each March.",
        "How much is the plot rent and when is it due?",
        "46 euros per year, due every March.",
    ),
    (
        "I put the compost bin in the south-east corner.",
        "The south-east corner gets some sun, which helps the compost warm up.",
        "Where is the compost bin?",
        "In the south-east corner.",
    ),
    (
        "The shed padlock code is 4-1-9-6.",
        "I'll remember the shed code 4-1-9-6.",
        "What is the shed padlock code?",
        "4-1-9-6.",
    ),
    (
        "My soil test came back with a pH of 6.4.",
        "A pH of 6.4 is close to ideal for most vegetables.",
        "What was the soil pH?",
        "6.4.",
    ),
    (
        "The allotment open day is on Saturday 21 June.",
        "21 June it is; the beds should look good by then.",
        "When is the allotment open day?",
        "Saturday 21 June.",
    ),
    (
        "I ordered netting from the club shop for 12 euros 50.",
        "12.50 for netting is fair; it keeps the birds off the lettuce.",
        "How much did the netting cost?",
        "12 euros 50.",
    ),
    (
        "The carrots went into row three, next to the onions.",
        "Carrots next to onions confuses the carrot fly; good choice.",
        "In which row are the carrots?",
        "Row three, next to the onions.",
    ),
    (
        "I plan to harvest the courgettes every two days once they start.",
        "Every two days keeps the courgettes small and tender.",
        "How often will the courgettes be harvested?",
        "Every two days once they start.",
    ),
]


def _short() -> Conversation:
    rng = random.Random(1)
    turns: list[Turn] = []
    questions: list[Question] = []
    for fact in GARDEN_FACTS:
        _plant(turns, questions, fact)
        turns.append(_filler(rng, "garden"))
    return Conversation(
        "short", "30 turns about one topic: planning an allotment garden.", turns, questions
    )


ARCHIVE_FACTS: list[Fact] = [
    (
        "The archive drive is labelled ORCHID-2 and holds 1.8 terabytes.",
        "ORCHID-2 with 1.8 TB, noted.",
        "What is the archive drive called and how much does it hold?",
        "ORCHID-2, 1.8 terabytes.",
    ),
    (
        "My aunt Mirela scanned the oldest prints in 2009.",
        "So the oldest scans are Mirela's work from 2009.",
        "Who scanned the oldest prints and when?",
        "Aunt Mirela, in 2009.",
    ),
    (
        "The wedding album has exactly 312 pictures.",
        "312 pictures in the wedding album.",
        "How many pictures are in the wedding album?",
        "312.",
    ),
    (
        "We agreed to name files as year-month-day followed by a place.",
        "Year-month-day plus place is a solid naming scheme.",
        "What naming scheme did we agree on for files?",
        "Year-month-day followed by a place.",
    ),
    (
        "The backup runs every Sunday at 22:00 to a second drive called FERN.",
        "Weekly backups on Sunday at 22:00 to FERN.",
        "When does the backup run and to which drive?",
        "Every Sunday at 22:00, to FERN.",
    ),
    (
        "The oldest photo in the archive is from 1958, taken at the harbour.",
        "A 1958 harbour photo is a real treasure.",
        "How old is the oldest photo and where was it taken?",
        "From 1958, at the harbour.",
    ),
    (
        "My cousin Bram wants copies of the school photos from 1987.",
        "I'll keep Bram's request for the 1987 school photos in mind.",
        "Who wants copies of which photos?",
        "Cousin Bram wants the 1987 school photos.",
    ),
    (
        "We decided to delete blurry duplicates only after a second check.",
        "Deleting only after a second check is the safe choice.",
        "What did we decide about blurry duplicates?",
        "Delete them only after a second check.",
    ),
    (
        "The scanner resolution we settled on is 1200 dpi.",
        "1200 dpi keeps enough detail for prints.",
        "Which scanner resolution was settled on?",
        "1200 dpi.",
    ),
    (
        "The shoebox of slides holds 64 slides from the 1970s.",
        "64 slides from the 1970s, that is a nice batch.",
        "How many slides are in the shoebox?",
        "64 slides, from the 1970s.",
    ),
    (
        "The family tree file is called roots-v3 and lives in the documents folder.",
        "roots-v3 in documents, noted.",
        "What is the family tree file called and where is it?",
        "roots-v3, in the documents folder.",
    ),
    (
        "I want the archive finished before my parents' anniversary on 9 October.",
        "9 October gives a clear deadline.",
        "What is the deadline for the archive?",
        "Before the parents' anniversary on 9 October.",
    ),
    (
        "The external scanner was borrowed from the library for 3 weeks.",
        "Three weeks with the library scanner; let's plan around that.",
        "How long was the scanner borrowed?",
        "For 3 weeks, from the library.",
    ),
    (
        "The winter folder turned out to contain 87 duplicates.",
        "87 duplicates in winter, I'll flag them.",
        "How many duplicates were in the winter folder?",
        "87.",
    ),
    (
        "We chose to share the archive with the family through a printed photo book of 40 pages.",
        "A 40-page photo book is a lovely way to share it.",
        "How will the archive be shared with the family?",
        "Through a printed photo book of 40 pages.",
    ),
]


def _tool_turn(rng: random.Random, n: int) -> Turn:
    """A turn whose tool result is a long listing (well above 2000 tokens)."""
    folder = rng.choice(WORDS["archive"])
    rows = "\n".join(
        f"IMG-{rng.randint(1000, 9999)} | {rng.randint(1958, 2024)}-"
        f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} | {rng.choice(WORDS['archive'])} | "
        f"{rng.randint(200, 9000) / 1000:.1f} MB | {rng.choice(['scan', 'camera', 'phone'])}"
        for _ in range(180)
    )
    call = f"call-{n}"
    return [
        {"role": "user", "content": f"Show me everything in the {folder} folder."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "list_folder", "args": {"folder": folder}, "id": call}],
        },
        {
            "role": "tool",
            "content": f"file | date | tag | size | source\n{rows}",
            "tool_call_id": call,
        },
        _answer(f"The {folder} folder has 180 files; most are scans."),
    ]


def _long() -> Conversation:
    rng = random.Random(2)
    turns: list[Turn] = []
    questions: list[Question] = []
    facts = iter(ARCHIVE_FACTS)
    fact_turns = {round(7 + i * 12.8) for i in range(len(ARCHIVE_FACTS))}
    while len(turns) < 200:
        n = len(turns) + 1
        if n in fact_turns:
            _plant(turns, questions, next(facts))
        elif n % 5 == 0:
            turns.append(_tool_turn(rng, n))
        else:
            turns.append(_filler(rng, "archive"))
    return Conversation(
        "long",
        "200 turns about a family photo archive, with large tool results (folder listings).",
        turns,
        questions,
    )


SWITCH_FACTS: dict[str, list[Fact]] = {
    "trip": [
        (
            "We booked the guesthouse Casa Velha in Valmora for 5 nights.",
            "Five nights at Casa Velha in Valmora, lovely.",
            "Where are we staying and for how long?",
            "Casa Velha in Valmora, for 5 nights.",
        ),
        (
            "Our train leaves on 3 May at 07:42 from platform 6.",
            "3 May, 07:42, platform 6. Set an alarm.",
            "When and from which platform does our train leave?",
            "On 3 May at 07:42, from platform 6.",
        ),
        (
            "My friend Ines Calder joins us on the third day.",
            "Ines Calder from day three, noted.",
            "Who joins on the third day?",
            "Ines Calder.",
        ),
        (
            "The ferry to the island costs 18 euros return.",
            "18 euros return for the ferry.",
            "How much does the ferry cost?",
            "18 euros return.",
        ),
        (
            "We decided to skip the cable car because of the queue.",
            "Skipping the cable car saves hours.",
            "Why did we skip the cable car?",
            "Because of the queue.",
        ),
    ],
    "piano": [
        (
            "My teacher, Mr Olsen, gave me the Minuet in G as my next piece.",
            "The Minuet in G is a classic next step.",
            "Which piece did the teacher assign, and who is the teacher?",
            "The Minuet in G, from Mr Olsen.",
        ),
        (
            "I practise on Tuesdays and Fridays from 19:30.",
            "Tuesdays and Fridays at 19:30, a good rhythm.",
            "When do I practise?",
            "Tuesdays and Fridays from 19:30.",
        ),
        (
            "My target tempo for the scales is 96 beats per minute.",
            "96 bpm is a good target for the scales.",
            "What is the target tempo for the scales?",
            "96 beats per minute.",
        ),
        (
            "The recital is in the community hall on 14 June.",
            "14 June in the community hall; plenty of time.",
            "Where and when is the recital?",
            "In the community hall on 14 June.",
        ),
        (
            "We agreed I play the waltz from memory at the recital.",
            "Playing the waltz from memory is a nice challenge.",
            "What did we agree about the waltz?",
            "I play it from memory at the recital.",
        ),
    ],
    "paint": [
        (
            "The wall colour is called Morning Fog, code MF-221.",
            "Morning Fog, MF-221, noted.",
            "Which wall colour was chosen?",
            "Morning Fog, code MF-221.",
        ),
        (
            "The living room wall is 4.2 metres wide and 2.6 metres high.",
            "So about 11 square metres per coat.",
            "How large is the living room wall?",
            "4.2 metres wide and 2.6 metres high.",
        ),
        (
            "I bought 2 tins of 2.5 litres each.",
            "Five litres should cover two coats.",
            "How much paint was bought?",
            "2 tins of 2.5 litres each.",
        ),
        (
            "My brother Teo helps me paint on Saturday.",
            "Teo on Saturday makes it a quicker job.",
            "Who helps paint, and when?",
            "Brother Teo, on Saturday.",
        ),
        (
            "We decided to paint the ceiling first and the wall second.",
            "Ceiling first is right: drips fall on the unpainted wall.",
            "In what order will the room be painted?",
            "The ceiling first, then the wall.",
        ),
    ],
}


def _switching() -> Conversation:
    rng = random.Random(3)
    turns: list[Turn] = []
    questions: list[Question] = []
    facts = {topic: iter(items) for topic, items in SWITCH_FACTS.items()}
    segments = [
        ("trip", 10),
        ("piano", 10),
        ("paint", 10),
        ("trip", 8),
        ("piano", 8),
        ("paint", 8),
        ("trip", 8),
        ("piano", 9),
        ("paint", 9),
    ]
    for topic, length in segments:
        for i in range(length):
            fact = next(facts[topic], None) if i in (1, 5) else None
            if fact is not None:
                _plant(turns, questions, fact)
            else:
                turns.append(_filler(rng, topic))
    return Conversation(
        "switching",
        "80 turns that switch between a trip, piano lessons and "
        "painting a room, returning to each topic.",
        turns,
        questions,
    )


def generate() -> list[Conversation]:
    return [_short(), _long(), _switching()]
