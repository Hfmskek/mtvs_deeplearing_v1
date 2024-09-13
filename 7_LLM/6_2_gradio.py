import gradio as gr

def chat(name, age, sex):
    return  f"이름은 : {name}, 나이는 : {age}, 성별은 {sex}", "안녕하세요"

app = gr.Interface(
    fn=chat,
    inputs=["textbox"]*3,
    outputs=['textbox', 'textbox']
)
app.launch()