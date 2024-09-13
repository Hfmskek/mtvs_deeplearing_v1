import gradio as gr

# 모델 가져오기
from langchain.chat_models import ChatOllama
model = ChatOllama(model="llama3.1:latest")


from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# system 메시지 구성
system_message = "당신은 {language} 선생님입니다. {language} 로 간단하게 답변해주세요"
system_prompt = SystemMessagePromptTemplate.from_template(system_message)

# human 메시지 구성
human_message = "{text} 에 대해서 얘기해줘"
human_prompt = HumanMessagePromptTemplate.from_template(human_message)

# 최종 프롬프트 구성
chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, 
     human_prompt]
    )

# 아웃풋 파서 설정
from langchain_core.output_parsers import StrOutputParser
outputparser = StrOutputParser()

chain = chat_prompt | model | outputparser



def greet(language, chat):
    result = chain.invoke({"language": language, "text": chat})
    return result

app = gr.Interface(
    fn=greet,
    inputs=["textbox", "textbox"],
    outputs="textbox"
    )

if __name__ == "__main__":
    app.launch(debug=True)