# Setup 1
with the setup:
    "llm_a": "gpt-4o-mini",
    "llm_o": "gpt-3.5-turbo",
    "llm_e": "gpt-4o",

I get a lot of errors:
- LLM-O often wrongly marks a solution as incorrect just because it has forbidden concepts. Even if the correct_solution is present in the solution from LLM-A
- LLM-O and LLM-E fail to produce a JSON output relatively often => see experiments\2_targeted_censorship\experiment_logs_20240725_182322\completion_reasons.png