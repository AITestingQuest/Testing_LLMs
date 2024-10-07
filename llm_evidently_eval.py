from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TextEvals
from evidently.descriptors import *
from evidently.metrics import *
from evidently.tests import *
from evidently.features.llm_judge import BinaryClassificationPromptTemplate
from ollama_llm_responses import execute_all_prompts

column_mapping = ColumnMapping(
    text_features=['Prompt', 'Response'],
)

verbose_judge = LLMEval(
    subcolumn="category",
    template = BinaryClassificationPromptTemplate(      
        criteria = """Conciseness refers to the quality of being brief and to the point, while still providing all necessary information.
            A concise response should:
            - Provide the necessary information without unnecessary details or repetition.
            - Be brief yet comprehensive enough to address the query.
            - Use simple and direct language to convey the message effectively.
        """,
        target_category="concise",
        non_target_category="verbose",
        uncertainty="unknown",
        include_reasoning=True,
        pre_messages=[("system", "You are a judge which evaluates text.")],
        ),
    provider = "openai",
    model = "gpt-4o"
)

factual_judge = LLMEval(
    subcolumn="category",
    template = BinaryClassificationPromptTemplate(      
        criteria = """The correct answer to the prompt should be factually correct.""",
        target_category="correct",
        non_target_category="incorrect",
        uncertainty="unknown",
        include_reasoning=True,
        pre_messages=[("system", "You are a judge which evaluates text.")],
        ),
    provider = "openai",
    model = "gpt-4o"
)

text_evals_report = Report(metrics=[
    TextEvals(column_name="Response",
              descriptors=[
                TextLength(),
                IncludesWords(
                    words_list=['delves','showcasing','underscores','comprehensive','pivotal','intricate'],
                    display_name = "LLM Gibberish"),
                Sentiment(),
                HuggingFaceToxicityModel(),
                DeclineLLMEval(),
                PIILLMEval(include_reasoning=True),
                verbose_judge,
                factual_judge
                ]
            ),
    ColumnSummaryMetric(column_name = SemanticSimilarity(
        display_name = "Response-Question Similarity"
    ).on(["Response", "Prompt"])
                        )
                    ]
                )

llms = ["phi3", "tinyllama", "stablelm2"]

for llm in llms:
    df = execute_all_prompts(llm)
    print(f"Prompts has been executed with {llm}")
    json_filename = f"result_evidently_{llm}.json"
    df.to_json(json_filename)
    print(f"Prompts and responses has been exported to {json_filename}")
    text_evals_report.run(reference_data=None,
                      current_data=df,
                      column_mapping=column_mapping)
    print(f"Evaluations has been executed for {llm}")
    report_json_filename = f"report_evidently_{llm}.json"
    text_evals_report.save_json(report_json_filename)
    print(f"Evaluation report has been exported to {report_json_filename}")
    report_html_filename = f"report_evidently_{llm}.html"
    text_evals_report.save_html(report_html_filename)
    print(f"Evaluation report has been exported to {report_html_filename}")
