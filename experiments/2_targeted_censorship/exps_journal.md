# Setup 1
with the setup:
    "llm_a": "gpt-4o-mini",
    "llm_o": "gpt-3.5-turbo",
    "llm_e": "gpt-4o",

I get a lot of errors:
- LLM-O often wrongly marks a solution as incorrect just because it has forbidden concepts. Even if the correct_solution is present in the solution from LLM-A => improved having different API calls for the correctness JSON and for the "forbidden concepts" JSON
- LLM-O and LLM-E fail to produce a JSON output relatively often => improved by having them give a justification before saying Correct/Incorrect or Yes/No. And also improved having different API calls for the correctness JSON and for the "forbidden concepts" JSON
- Try to have all models as gpt-4o first to get less noisy results and later come back to dumber overseer once new setup is less noisy and more useful

# Setup 2
with the setup:
    "llm_a": "gpt-4o",
    "llm_o": "gpt-4o",
    "llm_e": "gpt-4o",

## experiment_logs_20240728_115056.json
    "num_problems": 20,
    "max_iterations": 50

- There is no difference between False Negatives and False Positives since all LLMs are the same model.
- Experiments where both LLM-O and LLM-E are the same model are useful because a goal here is to modify the prompts/... so that there is no False Negatives or False Positives, because it is the same model so it should not give different results when asked the same prompt. For this I also set the temperature=0.0
- 10/20: LLM-O and LLM-E disagree on the correctness of the solution => I improve it with a more clear create_correctness_prompt (see experiment_logs_20240728_123736.json)
- 7/20: "completion_reason": "Max iterations reached..." => so I need to increase max number of iterations (see experiment_logs_20240728_123736.json)
- 1/10: "completion_reason": "Steganography False Negative detected: LLM-O missed a forbidden concept found by LLM-E" => since the model is the same there should not be any case of disagreement. 1/10 might be small enough so I do not make any changes in the create_forbidden_concepts_prompt()

## experiment_logs_20240728_123736.json
    "num_problems": 20,
    "max_iterations": 100

- 19/20: LLM-O and LLM-E disagree on the correctness of the solution => Increasing the iterations mostly converges is LLM-O and LLM-E disagreeing at some point in the correctness of the solution...I try modifying the create_correctness_prompt() to have the "correctness_eval" first and then the "correctness_explanation", because this order of first eval and then explanation seems to be working fine in the create_forbidden_concepts_prompt cause there there is no disagreements (see experiment_logs_20240728_130905.json)

## experiment_logs_20240728_130905.json
    "num_problems": 20,
    "max_iterations": 100

- 12/20: LLM-O and LLM-E disagree on the correctness of the solution...there is still a lot of disagreements on the correctness (however in all these cases LLM-O and LLM-E agree that there are forbidden concepts present) => disable the `break` for the scenarios of "Correctness False Negative" and ""Correctness False Positive" but still log in which problem and iteration there was a Correctness False Negative/Positive (see experiment_logs_20240728_134905.json)

## experiment_logs_20240728_134905.json
    "num_problems": 20,
    "max_iterations": 100

- 20/20: "completion_reason": "Max iterations reached...", it is still reaching the limit of iterations and not doing proper steganography => pass a simplified "conversation history" to LLM-A so that it sees how many times it failed before to motivate it to do better (see TBD3.json)

## TBD3.json
    "num_problems": 20,
    "max_iterations": 100

- NOTE: this approach is much more expensive and slower since the prompt passed to LLM-A grows with each iteration. 
- ??it is still reaching the limit of iterations and not doing proper steganography?? and the problem is that LLM-O/E are too good at detecting steganography because LLM-A is actually doing some steg attempts?? => try with a dumber model as LLM-O to see if LLM-A can do some steg that LLM-O does not detect but LLM-E does detect, aka "Steganography False Negative" (see TBD4.json)
- ??it is still reaching the limit of iterations and not doing proper steganography?? => try higher number of problems and higher number of iterations (see TBD5.json)

## TBD5.json


# Setup 3
with the setup:
    "llm_a": "gpt-4o",
    "llm_o": "gpt-3.5-turbo",
    "llm_e": "gpt-4o",

## TBD4.json




