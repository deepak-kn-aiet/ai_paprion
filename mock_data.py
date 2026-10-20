SCORE_COMPONENT_WEIGHTS = {
    "citation_impact": 0.40,
    "influential_citation_ratio": 0.20,
    "venue_quality": 0.20,
    "recency": 0.10,
    "review_score": 0.10,
}

SCORE_COMPONENT_LABELS = {
    "citation_impact": "Citation impact",
    "influential_citation_ratio": "Influential citation ratio",
    "venue_quality": "Venue quality",
    "recency": "Recency",
    "review_score": "Review score",
}

MOCK_RESULTS = [
    {
        "rank": 1,
        "title": "Attention Is All You Need",
        "authors": [
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
            "Jakob Uszkoreit",
            "Llion Jones",
            "Aidan N. Gomez",
            "Lukasz Kaiser",
            "Illia Polosukhin",
        ],
        "year": 2017,
        "venue": "NeurIPS",
        "citations": 148000,
        "influential_citations": 88000,
        "score": 0.97,
        "score_components": {
            "citation_impact": 0.97,
            "influential_citation_ratio": 0.96,
            "venue_quality": 0.98,
            "recency": 0.95,
            "review_score": 0.99,
        },
        "url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "summary": (
            "Introduces the Transformer architecture based solely on "
            "attention mechanisms, dispensing with recurrence and "
            "convolutions entirely."
        ),
        "overview": {
            "research_problem": (
                "Sequence transduction models relied on recurrent or "
                "convolutional layers, which limited parallelization "
                "and made long-range dependencies expensive to model."
            ),
            "main_contribution": (
                "Introduced the Transformer, an architecture built "
                "entirely on multi-head self-attention without "
                "recurrence or convolution."
            ),
            "related_topics": [
                "transformers",
                "attention mechanisms",
                "sequence modeling",
            ],
        },
        "verdict": {
            "recommendation": "Keep this paper",
            "explanation": (
                "Highest demonstration score in this result set "
                "(0.97 / 1.0), with strong citation impact and "
                "venue quality in the demonstration breakdown."
            ),
        },
    },
    {
        "rank": 2,
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2019,
        "venue": "NAACL",
        "citations": 98000,
        "influential_citations": 55000,
        "score": 0.93,
        "score_components": {
            "citation_impact": 0.94,
            "influential_citation_ratio": 0.92,
            "venue_quality": 0.95,
            "recency": 0.88,
            "review_score": 0.92,
        },
        "url": "https://arxiv.org/abs/1810.04805",
        "pdf_url": None,
        "summary": (
            "Pre-trains deep bidirectional representations from unlabeled "
            "text by jointly conditioning on left and right context."
        ),
        "overview": {
            "research_problem": (
                "Many language understanding benchmarks require deep "
                "bidirectional context, which earlier unidirectional "
                "language models could not capture during pre-training."
            ),
            "main_contribution": (
                "Pre-trained deep bidirectional representations by "
                "jointly conditioning on left and right context, then "
                "fine-tuned them for downstream tasks."
            ),
            "related_topics": [
                "language models",
                "pre-training",
                "text classification",
            ],
        },
        "verdict": {
            "recommendation": "Keep this paper",
            "explanation": (
                "Strong demonstration score (0.93 / 1.0), with "
                "high venue quality in the demonstration breakdown."
            ),
        },
    },
    {
        "rank": 3,
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": [
            "Patrick Lewis",
            "Ethan Perez",
            "Aleksandra Piktus",
            "Fabio Petroni",
            "Vladimir Karpukhin",
            "Naman Goyal",
            "Heinrich Kuttler",
            "Mike Lewis",
            "Wen-tau Yih",
            "Tim Rocktaschel",
            "Sebastian Riedel",
            "Douwe Kiela",
        ],
        "year": 2020,
        "venue": "NeurIPS",
        "citations": 8700,
        "influential_citations": 4100,
        "score": 0.88,
        "score_components": {
            "citation_impact": 0.89,
            "influential_citation_ratio": 0.86,
            "venue_quality": 0.90,
            "recency": 0.85,
            "review_score": 0.87,
        },
        "url": None,
        "pdf_url": None,
        "summary": (
            "Combines parametric sequence-to-sequence models with "
            "non-parametric dense passage retrieval for knowledge-intensive "
            "tasks."
        ),
        "overview": {
            "research_problem": (
                "Large language models struggle with knowledge-intensive "
                "tasks because their parametric knowledge is fixed after "
                "training and cannot be easily updated."
            ),
            "main_contribution": (
                "Combined parametric sequence-to-sequence generation with "
                "non-parametric dense passage retrieval to ground outputs "
                "in retrieved documents."
            ),
            "related_topics": [
                "retrieval-augmented generation",
                "question answering",
                "knowledge-intensive NLP",
            ],
        },
        "verdict": {
            "recommendation": "Consider another paper",
            "explanation": (
                "Lowest demonstration score in this result set "
                "(0.88 / 1.0); compare with the higher-ranked "
                "demonstration results before relying on it."
            ),
        },
    },
]


def get_mock_results():
    return MOCK_RESULTS
