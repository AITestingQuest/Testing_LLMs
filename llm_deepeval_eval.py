from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from ollama_llm_responses import execute_prompt

test_prompts = [
    'Classify this sentence as positive, negative or neutral: "I am very disappointed with the service I received at the restaurant"',
    'Select the location and the organization from that statement: "The United Nations is an international organization based in New York City."',
    'In the context of climate change, what is the greenhouse effect?',
    'Describe a day in the life of a time traveler who visits ancient Rome.',
    'Are the phrases "She enjoys hiking in the mountains" and "She loves trekking in the hills" similar in meaning?',
    'Correct the grammar in the sentence: "She don’t like the new movie."',
    'What is the result of 45 multiplied by 9?',
    'What is the result of this arithmetic expression: 2 + 2 * 2 + 2 * 5 - 4',
    'If all cats are animals and some animals are not dogs, can we conclude that some cats are not dogs?',
    'What is the capital of Paris?',
]

# Replace this with the actual documents that you are passing as input to your LLM.
contexts = [
        ["This statement is negative in sentiment."],
        ["The location is New York, the organization is the United Nations"],
        ["The greenhouse effect related to the greenhouse gases. They trap heat in the atmosphere, causing it to warm up the Earth's surface."],
        ["The actient Rome has famous places worth to visit: Forum Romanum, Colosseum, Pantheon"],
        ["The two phrases are similar in their meaning"],
        ["The correct grammar is: She doesn't like the new movie"],
        ["The result is 405"],
        ["The result is 12"],
        ["Yes, we can conclude that some cats are not dogs"],
        ["There is no such thing as capital of Paris, it is the capital city of France"]
]

# Replace this with the actual output from your LLM application
actual_outputs=[]
for test_prompt in test_prompts:
    actual_output = execute_prompt("phi3",test_prompt)
    print(actual_output)
    actual_outputs.append(actual_output)

test_case_names = ["sentiment", 
              "entity_recognition", 
              "general_knowledge", 
              "creativity",
              "semantic_similarity", 
              "grammar_correction", 
              "math_level1", 
              "math_level2", 
              "logical_reasoning", 
              "handling_misinformation"
              ]

test_cases = []

for i in range(10):
    test_case_names[i] = LLMTestCase(input=test_prompts[i], actual_output=actual_outputs[i], context=contexts[i])
    test_cases.append(test_case_names[i])

print(test_cases)

dataset = EvaluationDataset(test_cases=test_cases)

#metrics
hallucination_metric = HallucinationMetric(threshold=0.5)
answer_relevancy = AnswerRelevancyMetric()
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
overly_verbose = GEval(
    name="Overly verbose",
    criteria="Check whether the actual output overly verbose.",
    threshold=0.4,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

#evaluate
TestResult = evaluate(dataset, [hallucination_metric, answer_relevancy, correctness_metric, overly_verbose])

with open("test_result_phi3.csv", 'w', encoding="utf-8") as file:
        file.write("Prompt;Acutal Output;Test Result;Hallucination;Answer Relevancy;Correctness;Overly Verbose\n")
        for item in TestResult:
            test_input = item.input
            test_actual_output = item.actual_output
            test_success = item.success
            metric_success_text = ""
            metric_success = []
            for metrics_item in item.metrics_data:
                metric_success.append(metrics_item.success)
            for i in range(4):
                 metric_success_text = metric_success_text + str(metric_success[i]) + ";"
            file.write(f"{test_input};{test_actual_output};{test_success};{metric_success_text[:-1]}\n")
            
    

