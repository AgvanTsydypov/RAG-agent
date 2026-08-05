# Small labeled evaluation set. Grow it as new failure modes appear.
# Quantitative items intentionally stress the baseline (it has no tools for math).
EVAL_SET = [
    {
        "question": "What supply chain risks does Apple describe in its 10-K?",
        "reference": (
            "Apple depends on outsourcing partners, many outside the U.S., for "
            "component manufacturing, final assembly, and logistics, and relies on "
            "single-source custom components, creating supply shortage and pricing "
            "risk. It is also exposed to disruption from natural disasters, "
            "geopolitical events, and industrial accidents at suppliers."
        ),
    },
    {
        "question": "How did Apple's net sales change from FY2023 to FY2024, in percent?",
        "reference": (
            "Apple's net sales rose from about $383.285 billion in FY2023 to about "
            "$391.035 billion in FY2024, an increase of roughly 2.0%."
        ),
    },
    {
        "question": "How did Apple's net sales change from FY2024 to FY2025, in percent?",
        "reference": (
            "Apple's net sales rose from about $391.035 billion in FY2024 to about "
            "$416.161 billion in FY2025, an increase of roughly 6.4%."
        ),
    },
    {
        "question": "How did Apple's iPhone revenue trend from FY2023 to FY2025?",
        "reference": (
            "Apple's iPhone revenue grew from about $200.6 billion in FY2023 to "
            "about $201.2 billion in FY2024 and about $209.6 billion in FY2025."
        ),
    },
    {
        "question": "What risk factors does Microsoft describe about its business?",
        "reference": (
            "Microsoft describes risks such as intense competition, cybersecurity "
            "and data-security threats, risks tied to its AI investments, legal and "
            "regulatory scrutiny, and reliance on its cloud and datacenter operations."
        ),
    },
    {
        "question": "What was Apple's total number of retail employees in 2015?",
        "reference": "The provided 10-K filings do not contain this information.",
    },
]