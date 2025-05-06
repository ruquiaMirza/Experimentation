from typing import Any

from pydantic import validate_call

from distilabel.llms import AsyncLLM, LLM, VertexAILLM
from distilabel.typing import GenerateOutput, HiddenState
from distilabel.typing.base import ChatType
from distilabel.steps.tasks import TextGeneration
from gemini_connect import  GeminiCustomGenerativeAI
from pydantic import BaseModel, Field, validate_call
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.steps.base import StepInput


math_topics = [
    "Algebraic Expressions",
    "Linear Equations",
    "Quadratic Equations",
    "Polynomial Functions"
]

dataset2 = [{"topic": topic} for topic in math_topics]


class CustomAsyncLLM(AsyncLLM):
    """
    A custom asynchronous LLM for use with distilabel, integrating Gemini.
    """

    gemini_llm: GeminiCustomGenerativeAI = Field(default_factory=GeminiCustomGenerativeAI, description="The Gemini LLM instance.")

    @property
    def model_name(self) -> str:
        return "gemini-2.0-flash"  # Or whatever model you're using

    async def agenerate(self, input, num_generations=1, **kwargs: Any):
        """
        Asynchronously generates content using the Gemini LLM.

        Args:
            inputs: A list of input prompts.
            num_generations: The number of generations to produce for each prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of lists, where each inner list contains the generated content for a given prompt.
        """


        results = []
        for input_obj in input:
            print(f"input+obj{input_obj}")
            # Check if 'instruction' key exists, otherwise try 'input'
            prompt = input_obj.get("content") or input_obj.get("instruction")
            print(prompt)
            if not prompt:
                print(f"Warning: No 'instruction' or 'input' key found in input object: {input_obj}")
                results.append([""])  # Append an empty string if no prompt is found
                continue

            generations = []
            for _ in range(num_generations):
                try:
                    generation = self.gemini_llm.generate_content(prompt=prompt)
                    generations.append(generation)
                except Exception as e:
                    generations.append(f"Error generating content: {str(e)}")  # Handle errors
            results.append(generations)
        return {"generations": results,"statistics": {}}


def get_last_hidden_state(self, inputs):
        """
        Placeholder for getting the last hidden state.  Not implemented in this example.
        """
        pass

if __name__ == '__main__':
    cllm = CustomAsyncLLM()
    #With dataset2
    with Pipeline(name="text_generation_dataset2") as pipeline_text_generation2:
      load_data = LoadDataFromDicts(
          name="load_data",
          data=dataset2,
          output_mappings={"topic": "instruction"},
      )
      print(load_data)
      text_generation = TextGeneration(name="text_generation", llm=cllm)
      keep_columns = KeepColumns(
            name="keep_columns",
            columns=[
                "instruction",
                "generation",
            ],
        )


      load_data >> text_generation >> keep_columns
    distiset_text_generation2 = pipeline_text_generation2.run(use_cache=False,parameters={"text_generation": {"llm":  {"kwargs": {"temperature": 0.7}}}})

    print(distiset_text_generation2["default"]["train"][:])