from typing import Any

from pydantic import validate_call

from distilabel.llms import AsyncLLM, LLM, VertexAILLM
from distilabel.typing import GenerateOutput, HiddenState
from distilabel.typing.base import ChatType
from distilabel.steps.tasks import TextGeneration
from gemini_connect import  GeminiCustomGenerativeAI
from pydantic import BaseModel, Field, validate_call
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns,CombineOutputs, ExpandColumns
from distilabel.steps.tasks import TextGeneration, APIGenGenerator, Genstruct, SelfInstruct,EvolInstruct, UltraFeedback
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



        for input_obj in input:
            #print(f"input+obj{input_obj}")
            # Check if 'instruction' key exists, otherwise try 'input'
            prompt = input_obj.get("content") or input_obj.get("instruction")
            #print(prompt)
            if not prompt:
                print(f"Warning: No 'instruction' or 'input' key found in input object: {input_obj}")
                generations=[]  # Append an empty string if no prompt is found
                continue

            generations = []
            stats=[]
            for _ in range(num_generations):
                try:
                    generation = self.gemini_llm.generate_content(prompt=prompt)
                    print(generation)
                    generations.append(generation['generations'])
                    stats=generation['stats']
                except Exception as e:
                    generations.append(f"Error generating content: {str(e)}")  # Handle errors

            print(f"generations:{generations}")
            print(f"stats:{stats}")

        return {"generations": generations,"statistics": stats}


    def get_last_hidden_state(self, inputs):
            """
            Placeholder for getting the last hidden state.  Not implemented in this example.
            """
            pass

if __name__ == '__main__':
    cllm = CustomAsyncLLM()


    with Pipeline(name="text_generation_dataset2") as pipeline_text_generation2:
      load_data = LoadDataFromDicts(
          name="load_data",
          data=[{"topic":"The dough will be soft and the chocolate chips may not stick because of the melted butter. Just keep stirring it; I promise it will come together. Because of the melted butter and extra egg yolk, the slick dough doesn’t even look like normal cookie dough! Trust the process…"}],
          output_mappings={"topic": "input"},
      )
      selfinstruct=SelfInstruct(name='self_instruct',llm=cllm,output_mappings={"instructions":"instruction"})
      # load_data1 = LoadDataFromDicts(
      #     name="load_data1",
      #     data=[{
      #               "topic": "The dough will be soft and the chocolate chips may not stick because of the melted butter. Just keep stirring it; I promise it will come together. Because of the melted butter and extra egg yolk, the slick dough doesn’t even look like normal cookie dough! Trust the process…"}],
      #     output_mappings={"topic": "generations"},
      # )
      #expand=ExpandColumns(columns=['instruction'])
      combine=CombineOutputs()
      #ultrafeedback=UltraFeedback(name='ultra_feedback',llm=cllm)



      #evol_instruct=EvolInstruct(name='evolinstruct',llm=cllm,num_evolutions=2, generate_answers=True,include_original_instruction=True)
      #magpie=SelfInstruct(name='magpie_Gen', llm=cllm,num_instructions=4)
      #text_generation = TextGeneration(name="text_generation", llm=cllm)
      #genstruct=Genstruct(name="Gen_struct",llm=cllm)
      keep_columns = KeepColumns(
            name="keep_columns",
            columns=[
                "instruction",
                "input",

            ],
           output_mappings={"input": "generations"},
        )

      #evol_instruct=EvolInstruct(name='evolinstruct',llm=cllm,num_evolutions=1)

      [load_data >> selfinstruct] >> combine >> keep_columns
      distiset_text_generation2 = pipeline_text_generation2.run(use_cache=False, parameters={"self_instruct": {"llm": {"kwargs": {"temperature": 0.7}}},"evolinstruct": {"llm": {"kwargs": {"temperature": 0.7}}},"ultra_feedback": {"llm": {"kwargs": {"temperature": 0.7}}}})

      result=distiset_text_generation2["default"]["train"][:]
      print(result)
      # print(result["input"])
      # for x in [result["instructions"][0]]:
      #     for y in x:
      #         print(y)

