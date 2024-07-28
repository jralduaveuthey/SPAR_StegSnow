# Setup 1
with the setup:
    "llm_a": "gpt-4o-mini",
    "llm_o": "gpt-3.5-turbo",
    "llm_e": "gpt-4o",

I get a lot of errors:
- LLM-O often wrongly marks a solution as incorrect just because it has forbidden concepts. Even if the correct_solution is present in the solution from LLM-A => improved having different API calls for the correctness JSON and for the "forbidden concepts" JSON
- LLM-O and LLM-E fail to produce a JSON output relatively often => improved by having them give a justification before saying Correct/Incorrect or Yes/No. And also improved having different API calls for the correctness JSON and for the "forbidden concepts" JSON

# Setup 2
with the setup:
    "llm_a": "gpt-4o",
    "llm_o": "gpt-4o",
    "llm_e": "gpt-4o",

## experiment_logs_20240728_115056.json
- 7/20 x "completion_reason": "Max iterations reached. LLM-O thinks the final solution is correct, but there are forbidden concepts." => so I need to increase interat
