############ Gradio + Ray Serve ############
import requests
from ray import serve
from ray.serve.gradio_integrations import GradioServer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from transformers import pipeline


example_input = (
    '["Hello!", "Hi!", "I really love your wife", "I wanna kill you"]'
)


def gradio_classifier_builder():

    classifier = pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis"
        )

    def pie_chart(p1_emotions: list, p2_emotions: list):
        labels = ["Positive", "Neutral", "Negative"]

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'},
                                                    {'type': 'domain'}]],
                            subplot_titles=['First person', 'Second person'])
        fig.add_trace(go.Pie(labels=labels, values=p1_emotions,
                             name="Person 1"), 1, 1)
        fig.add_trace(go.Pie(labels=labels, values=p2_emotions,
                             name="Person 2"), 1, 2)

        fig.update_traces(hoverinfo="label+percent+name")
        fig.update_layout(title_text='Emotions in dialog')

        return fig

    def classify_batched(text: str):
        batched_inputs = eval(text)
        results = classifier(batched_inputs)
        results = [result["label"] for result in results]
        k = 1
        first_person = {'POS': 0, 'NEU': 0, 'NEG': 0}
        second_person = {'POS': 0, 'NEU': 0, 'NEG': 0}
        for result in results:
            if k % 2 != 0:
                first_person[result] += 1
                k += 1
            else:
                second_person[result] += 1
                k += 1
        p1_emotions = list(first_person.values())
        p2_emotions = list(second_person.values())
        chart = pie_chart(p1_emotions, p2_emotions)

        return chart

    return gr.Interface(
        fn=classify_batched,
        inputs=[gr.Textbox(value=example_input,
                           label="Input replica",
                           lines=5)],
        outputs=gr.Plot()
    )


app = GradioServer.options(ray_actor_options={"num_cpus": 4}).bind(
    gradio_classifier_builder
)

serve.run(app)
# ! serve run app:app

############################## Fastapi + Ray #################################
# from ray import serve
# from fastapi import FastAPI
# import requests
# from transformers import pipeline
# from typing import List

# app = FastAPI()

# @serve.deployment
# @serve.ingress(app)
# class SentimentAnalysis:
#     def __init__(self):
#         self._classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

#     @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
#     async def classify_batched(self, batched_inputs):
#         print("Got batch size:", len(batched_inputs))
#         results = self._classifier(batched_inputs)
#         return [result["label"] for result in results]

#     @app.get("/")
#     async def classify(self, input_text: str) -> str:
#         return await self.classify_batched(input_text)


# serve.run(SentimentAnalysis.bind())

# text = [
#     "Fuck you"
# ]

# print(requests.get("http://localhost:8000/", params={"input_text": text}).json())
