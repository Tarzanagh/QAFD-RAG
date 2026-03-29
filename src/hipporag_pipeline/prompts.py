"""
Inline prompt templates for NER, triple extraction, query NER,
fact reranking, and RAG QA — adapted from HippoRAG's template files.
"""

from string import Template
from typing import List, Dict

# ============================================================================
# NER (passage → named entities)
# ============================================================================

NER_SYSTEM = (
    "Your task is to extract named entities from the given paragraph. "
    "Respond with a JSON list of entities."
)

NER_ONE_SHOT_INPUT = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music "
    "portal - PlanetRadiocity.com that offers music related news, videos, songs, and "
    "other music-related features."
)

NER_ONE_SHOT_OUTPUT = (
    '{"named_entities":\n'
    '    ["Radio City", "India", "3 July 2001", "Hindi", "English", '
    '"May 2008", "PlanetRadiocity.com"]\n'
    '}'
)

def make_ner_messages(passage: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user",   "content": NER_ONE_SHOT_INPUT},
        {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
        {"role": "user",   "content": passage},
    ]

# ============================================================================
# Query NER (question → named entities)
# ============================================================================

QUERY_NER_SYSTEM = "You're a very effective entity extraction system."

QUERY_NER_ONE_SHOT_INPUT = (
    "Please extract all named entities that are important for solving the questions below.\n"
    "Place the named entities in json format.\n\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
)

QUERY_NER_ONE_SHOT_OUTPUT = (
    '\n{"named_entities": ["First for Women", "Arthur\'s Magazine"]}\n'
)

def make_query_ner_messages(query: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_NER_SYSTEM},
        {"role": "user",   "content": QUERY_NER_ONE_SHOT_INPUT},
        {"role": "assistant", "content": QUERY_NER_ONE_SHOT_OUTPUT},
        {"role": "user",   "content": f"Question: {query}"},
    ]

# ============================================================================
# Triple extraction (passage + entities → RDF triples)
# ============================================================================

TRIPLE_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from "
    "the given passages and named entity lists. "
    "Respond with a JSON list of triples, with each triple representing a relationship "
    "in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities "
    "in the list for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n"
)

_TRIPLE_FRAME = (
    "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
    "Paragraph:\n```\n{passage}\n```\n\n{named_entity_json}\n"
)

TRIPLE_ONE_SHOT_INPUT = _TRIPLE_FRAME.format(
    passage=NER_ONE_SHOT_INPUT,
    named_entity_json=NER_ONE_SHOT_OUTPUT,
)

TRIPLE_ONE_SHOT_OUTPUT = (
    '{"triples": [\n'
    '            ["Radio City", "located in", "India"],\n'
    '            ["Radio City", "is", "private FM radio station"],\n'
    '            ["Radio City", "started on", "3 July 2001"],\n'
    '            ["Radio City", "plays songs in", "Hindi"],\n'
    '            ["Radio City", "plays songs in", "English"],\n'
    '            ["Radio City", "forayed into", "New Media"],\n'
    '            ["Radio City", "launched", "PlanetRadiocity.com"],\n'
    '            ["PlanetRadiocity.com", "launched in", "May 2008"],\n'
    '            ["PlanetRadiocity.com", "is", "music portal"],\n'
    '            ["PlanetRadiocity.com", "offers", "news"],\n'
    '            ["PlanetRadiocity.com", "offers", "videos"],\n'
    '            ["PlanetRadiocity.com", "offers", "songs"]\n'
    '    ]\n'
    '}'
)


def make_triple_messages(passage: str, named_entities: List[str]) -> List[Dict[str, str]]:
    import json as _json
    named_entity_json = _json.dumps({"named_entities": named_entities})
    user_content = _TRIPLE_FRAME.format(passage=passage, named_entity_json=named_entity_json)
    return [
        {"role": "system", "content": TRIPLE_SYSTEM},
        {"role": "user",   "content": TRIPLE_ONE_SHOT_INPUT},
        {"role": "assistant", "content": TRIPLE_ONE_SHOT_OUTPUT},
        {"role": "user",   "content": user_content},
    ]

# ============================================================================
# Fact reranker / filter  (DSPy-style prompt from HippoRAG)
# ============================================================================

RERANKER_SYSTEM = (
    "Your input fields are:\n"
    "1. `question` (str): Query for retrieval\n"
    "2. `fact_before_filter` (str): Candidate facts to be filtered\n\n"
    "Your output fields are:\n"
    '1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\n'
    "All interactions will be structured in the following way, with the appropriate "
    "values filled in.\n\n"
    "[[ ## question ## ]]\n{question}\n\n"
    "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "[[ ## fact_after_filter ## ]]\n{fact_after_filter}        "
    '# note: the value you produce must be pareseable according to the following JSON schema: '
    '{"type": "object", "properties": {"fact": {"type": "array", '
    '"description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", '
    '"items": {"type": "array", "items": {"type": "string"}}, '
    '"title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n\n'
    "[[ ## completed ## ]]\n\n"
    "In adhering to this structure, your objective is: \n"
    "        You are a critical component of a high-stakes question-answering system used by "
    "top researchers and decision-makers worldwide. Your task is to filter facts based on their "
    "relevance to a given query, ensuring that the most crucial information is presented to "
    "these stakeholders. The query requires careful analysis and possibly multi-hop reasoning "
    "to connect different pieces of information. You must select up to 4 relevant facts from "
    "the provided candidate list that have a strong connection to the query, aiding in reasoning "
    "and providing an accurate answer. The output should be in JSON format, e.g., "
    '{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, '
    'return an empty list, {"fact": []}. The accuracy of your response is paramount, as it '
    "will directly impact the decisions made by these high-level stakeholders. You must only "
    "use facts from the candidate list and not generate new facts. The future of critical "
    "decision-making relies on your ability to accurately filter and present relevant information."
)

RERANKER_INPUT_TEMPLATE = (
    "[[ ## question ## ]]\n{question}\n\n"
    "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "Respond with the corresponding output fields, starting with the field "
    "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)

RERANKER_OUTPUT_TEMPLATE = (
    "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n"
    "[[ ## completed ## ]]"
)

# Built-in few-shot demos (from HippoRAG's filter_default_prompt.py)
RERANKER_DEMOS = [
    {
        "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
        "fact_before_filter": '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}',
        "fact_after_filter": '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}',
    },
    {
        "question": "When is the director of film The Ancestor 's birthday?",
        "fact_before_filter": '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
        "fact_after_filter": '{"fact":[["the ancestor","directed by","guido brignone"]]}',
    },
    {
        "question": "In what geographic region is the country where Teafuone is located?",
        "fact_before_filter": '{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}',
        "fact_after_filter": '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}',
    },
    {
        "question": "When did the director of film S.O.B. (Film) die?",
        "fact_before_filter": '{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}',
        "fact_after_filter": '{"fact":[["s o b","written and directed by","blake edwards"]]}',
    },
]


def make_reranker_messages(question: str, fact_before_filter_json: str) -> List[Dict[str, str]]:
    """Build the full chat history for fact reranking."""
    messages = [{"role": "system", "content": RERANKER_SYSTEM}]
    for demo in RERANKER_DEMOS:
        messages.append({
            "role": "user",
            "content": RERANKER_INPUT_TEMPLATE.format(
                question=demo["question"],
                fact_before_filter=demo["fact_before_filter"],
            ),
        })
        messages.append({
            "role": "assistant",
            "content": RERANKER_OUTPUT_TEMPLATE.format(
                fact_after_filter=demo["fact_after_filter"],
            ),
        })
    messages.append({
        "role": "user",
        "content": RERANKER_INPUT_TEMPLATE.format(
            question=question,
            fact_before_filter=fact_before_filter_json,
        ),
    })
    return messages

# ============================================================================
# RAG QA prompt (MuSiQue-style, also used for HotpotQA / 2Wiki)
# ============================================================================

RAG_QA_SYSTEM = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages "
    "and corresponding questions meticulously. "
    "Your response start after \"Thought: \", where you will methodically break down the "
    "reasoning process, illustrating how you arrive at conclusions. "
    "Conclude with \"Answer: \" to present a concise, definitive response, devoid of "
    "additional elaborations."
)

_RAG_QA_ONE_SHOT_DOCS = (
    "Wikipedia Title: The Last Horse\n"
    "The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed "
    "by Edgar Neville starring Fernando Fernán Gómez.\n\n"
    "Wikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its Royal "
    "Charter as a university in 1952, has over 22,000 students. The university is ranked "
    "in the top 100 research universities in the world in the Academic Ranking of World "
    "Universities 2010.\n\n"
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the "
    "University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered "
    "Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF).\n"
)

RAG_QA_ONE_SHOT_INPUT = (
    f"{_RAG_QA_ONE_SHOT_DOCS}\n\n"
    "Question: When was Neville A. Stanton's employer founded?\nThought: "
)

RAG_QA_ONE_SHOT_OUTPUT = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


def make_qa_messages(passages: List[str], question: str) -> List[Dict[str, str]]:
    """Build QA chat messages from retrieved passages and question."""
    prompt_user = ""
    for p in passages:
        prompt_user += f"Wikipedia Title: {p}\n\n"
    prompt_user += f"Question: {question}\nThought: "
    return [
        {"role": "system", "content": RAG_QA_SYSTEM},
        {"role": "user",   "content": RAG_QA_ONE_SHOT_INPUT},
        {"role": "assistant", "content": RAG_QA_ONE_SHOT_OUTPUT},
        {"role": "user",   "content": prompt_user},
    ]

# ============================================================================
# Query instruction strings (for embedding model)
# ============================================================================

QUERY_INSTRUCTIONS = {
    "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
    "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
    "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
    "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
    "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
}

def get_query_instruction(linking_method: str) -> str:
    default = "Given a question, retrieve relevant documents that best answer the question."
    return QUERY_INSTRUCTIONS.get(linking_method, default)
