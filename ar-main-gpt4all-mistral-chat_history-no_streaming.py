from deep_translator import GoogleTranslator
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download

from langchain_community.llms import LlamaCpp
import gpt4all
from gpt4all import GPT4All
from deep_translator import GoogleTranslator
import asyncio
import sys
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="" , token_no = 0):
        self.container = container
        self.text = initial_text
        self.token_no = token_no

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        #self.text = ""
        self.token_no = self.token_no+1
        # Add the response to the chat window
##        with self.container.empty():
##            st.markdown(self.text).clear()
        with self.container.empty():
            self.container.header("Chat Session")
            with self.container.chat_message("ai"):
                print(str(self.token_no))
                if self.token_no != 0:
                    self.text += token
                    st.markdown(self.text)
                else:
                    self.text = ""
        #self.container.markdown(self.text)


@st.cache_resource
def create_chain(system_prompt):
    model = GPT4All(device = 'gpu' , model_name = "alaa_ai_model_mistral_v1.9.gguf" , model_path ='C:/Users/Alaa AI/Python Projects/Ai Models/' , allow_download = False)
    st.markdown("Model Loaded Successfully....")
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    #stream_handler = StreamHandler(st.sidebar.empty())
    #stream_handler = StreamHandler(st.chat_message("ai").empty())
    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear that the LLM is typing the
    # responses in real-time.
    #callback_manager = CallbackManager([stream_handler])

##    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
##                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")

##    model_path = hf_hub_download(repo_id=repo_id,
##                                 filename=model_file_name,
##                                 repo_type="model")

    # initialize LlamaCpp LLM model
    # n_gpu_layers, n_batch, and n_ctx are for GPU support.
    # When not set, CPU will be used.
    # set 1 for Mac m2, and higher numbers based on your GPU support
##    llm = LlamaCpp(
##            model_path="C:/Users/Alaa AI/Python Projects/Ai Models/llama-2-7b-chat.q4_K_M.gguf",            temperature=0,
##            max_tokens=512,
##            top_p=1,
##            # callback_manager=callback_manager,
##            # n_gpu_layers=1,
##            # n_batch=512,
##            # n_ctx=4096,
##            stop=["[INST]"],
##            verbose=False,
##            streaming=True,
##            )
##    template_messages = [
##        SystemMessage(content="You are a helpful assistant."),
##        MessagesPlaceholder(variable_name="chat_history"),
##        HumanMessagePromptTemplate.from_template("{text}"),
##    ]
    
##    llm = LlamaCpp(
##        model_path="C:/Users/Alaa AI/Python Projects/Ai Models/llama-2-7b-chat.q4_K_M.gguf",
##        #streaming=False, n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True
##         n_gpu_layers = 15000 , n_ctx = 5120 , streaming=True ,max_tokens = 5000 ,n_batch=512,verbose = False, use_mlock = True , use_mmap = True
##    )
    #prompt_template = ChatPromptTemplate.from_messages(template_messages)
    #model = Llama2Chat(llm=llm , callback_manager = callback_manager)
    # Template you will use to structure your user input before converting
    # into a prompt. Here, my template first injects the personality I wish to
    # give to the LLM before in the form of system_prompt pushing the actual
    # prompt from the user. Note that this chatbot doesn't have any memory of
    # the conversation. So we will inject the system prompt for each message.
##    template = """
##    <s>[INST]{}[/INST]</s>
##
##    [INST]{}[/INST]
##    """.format(system_prompt, "{question}")
##
##    # We create a prompt from the template so we can use it with Langchain
##    prompt = PromptTemplate(template=template, input_variables=["question"])

    # We create an llm chain with our LLM and prompt
    # llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy
##    llm_chain = prompt | llm  # LCEL
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #llm_chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
    #memory.clear()
    return model


# Set the webpage title
st.set_page_config(
    page_title="Alaa's Chat Robot!"
)

# Create a header element
st.header("Alaa's Chat Robot!")



# Create Select Box
lang_opts = ["ar","en" , "fr"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)
# This sets the LLM's personality for each prompt.
# The initial personality provided is basic.
# Try something interesting and notice how the LLM responses are affected.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")


# Create LLM chain to use for our chatbot.
mod = create_chain(system_prompt)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def disable():
    st.session_state.disabled = True
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False
    
with mod.chat_session():
# We take questions/instructions from the chat input to pass to the LLM
    if user_prompt := st.chat_input("Your message here", key="user_input" , on_submit = disable , disabled=st.session_state.disabled):
        del st.session_state.disabled
        if "disabled" not in st.session_state:
            st.session_state.disabled = False
        #st.chat_input("Your message here", key="disabled_chat_input", disabled=True)
        st.markdown("in session")
        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        
        user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
        
        
##
##            # Pass our input to the LLM chain and capture the final responses.
##            # It is worth noting that the Stream Handler is already receiving the
##            # streaming response as the llm is generating. We get our response
##            # here once the LLM has finished generating the complete response.
##            
        response = mod.generate(prompt = user_prompt , max_tokens = 50)
        response = GoogleTranslator(source='auto', target=lang_selected).translate(response)
##            # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
##
        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)
        st.rerun()

