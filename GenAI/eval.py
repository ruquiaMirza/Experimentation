import vertexai
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import GeminiModel
from vertexai.generative_models import GenerativeModel, GenerationResponse
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import pandas as pd




class EvalGoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model =  GenerativeModel(model)
        print(self.model)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        result= self.model.generate_content(contents=prompt)
        return result.candidates[0].content.parts[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Vertex AI Model"



#TODO : Add values for project and location below
project_id='gen-lang-client-0667165687'
vertexai.init(project=project_id, location='us-central1')

cmodel = GeminiModel(
    model_name="gemini-2.0-flash",
    project=project_id,
    location="us-central1",
    temperature=0.7
)

# initialize the  wrapper class
vertexai_gemini = EvalGoogleVertexAI("gemini-2.0-flash")
print(cmodel.generate("Write me a joke"))


# metric = AnswerRelevancyMetric(
#     threshold=0.7,
#     model=vertexai_gemini,
#     include_reason=True
# )
inputs=["Analyze how the incorporation of melted butter affects the adherence of chocolate chips in dough.",
"How does incorporating melted butter and an extra egg yolk affect dough consistency and chocolate chip adhesion?",
"How does melted butter affect cookie dough consistency and assembly?",
"How should I continuously combine slick cookie dough with melted butter and additional egg yolk?",
"Examine the role of emulsified fats in altering cookie dough's texture and cohesion dynamics."
]
#inputs=["Explain why melted butter might affect the dough's consistency in this recipe.", 'Describe how stirring helps the dough come together despite its slick texture.', 'Should I refrigerate the dough to firm it up, or is baking straight away recommended?', 'Summarize the key indicators that the cookie dough will eventually reach the correct state.', 'List potential adjustments to the recipe to avoid melted butter issues in the future.']
actual_output='The dough will be soft and the chocolate chips may not stick because of the melted butter. Just keep stirring it; I promise it will come together. Because of the melted butter and extra egg yolk, the slick dough doesn’t even look like normal cookie dough! Trust the process…'

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the queries in 'input' are the right set of questions to 'actual output'",
        "Given the queries in 'input' the answer obtianed through 'actual output' is the correct answer without any vague results"

    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=vertexai_gemini,
)

for x in inputs:

    test_case = LLMTestCase(
        input=x,
        # Replace this with the output from your LLM app
        actual_output=actual_output,
        #retrieval_context=retrieval_context

    )
    evaluate(test_cases=[test_case], metrics=[correctness])
