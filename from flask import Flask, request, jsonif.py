
from flask import Flask, request, jsonify
import os
import tempfile
from langgraph.graph import Graph, StateGraph, END
import google.generativeai as genai
from PIL import Image
import requests
from typing import Dict, Any, Optional, TypedDict, Literal
import dotenv
import os
api_key = os.getenv('GOOGLE_API')  # , 'your_google_api_key'
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load configuration from .env file
dotenv.load_dotenv()

# Configure Generative AI
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Configure Google Search API

cx = os.getenv('GOOGLE_CX')  # , 'your_google_search_cx'
def google_search(api_key: str, cx: str, query: str) -> Dict[str, Any]:
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cx,
        'q': query
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
class GraphState(TypedDict):
    user_input: Optional[list[str]]
    photo_description: Optional[str]
    feedback: Optional[str]
    llm_output: Optional[str]
    image_path: Optional[str]
    chat_history: Optional[list[str]]

def get_last_two_chats(chat_history: list[str]) -> str:
    his=" ".join(chat_history[-4:]) if len(chat_history) >= 4 else " ".join(chat_history)
    return his

def first_user_input(state: GraphState) -> GraphState:
    if state.get("chat_history") is None:
        state["chat_history"] = []
    return state

def decision(state: GraphState) -> Literal["Take a photo", "Fetch details from the image", "Save the photo", "Take a video", "Error", "Use Chat bot", 'Search the web for particular detail']:
    last_two_chats = get_last_two_chats(state["chat_history"])
    prompt = f"""You are a decision making intelligent machine inside the pipeline of an app which helps the visually impaired. Use the internet for extra information. Only take a photo if the user clearly asks for the same.The user is querying about the description of the photo or Chatting. Your task is to decide what the next step is by selecting what the output string is according to the user input. Output 'Fetch details from the image' if no user_input is found. Redirect all innappropriate comments to the chatbot. Output the single string without quotes. Only output the exact corresponding string chosen from the following strings:
                'Take a photo', 'Fetch details from the image', 'Save the photo', 'Take a video', 'Search the web for particular detail', 'Use Chat bot' and 'Error'.
                This is the user's input: {state["user_input"][-1]}
                Recent chat history: {last_two_chats}"""
    response = model.generate_content(prompt)
    state["llm_output"] = response.text.strip()

    valid_outputs = {"Take a photo", "Fetch details from the image", "Save the photo", "Take a video", "Error", "Use Chat bot", 'Search the web for particular detail'}

    if state["llm_output"] in valid_outputs:
        return state["llm_output"]
    else:
        state["llm_output"] = "Error"
        return state["llm_output"]

def take_photo(state: GraphState) -> GraphState:
    state["llm_output"] = "Photo taken"
    return state
Base_prompt="""You are a helpful and attentive AI assistant designed to describe photos for blind users. Your primary goal is to provide vivid, detailed descriptions that help users "see" through your words. Focus on elements that convey the overall atmosphere, mood, and context of the image. Describe colors, textures, spatial relationships, and any interesting or unusual details. Be specific about the positioning of objects and people. Mention facial expressions, body language, and clothing when relevant. For outdoor scenes, describe the weather, time of day, and natural elements. For indoor photos, explain the type of room and its furnishings. Always ask if the user would like more details about any particular aspect of the image. Be patient and willing to repeat or clarify information. Offer to break down complex images into smaller parts for easier understanding. Use descriptive language that appeals to other senses, like describing how things might feel or smell. Your tone should be warm, engaging, and encouraging, making the experience of "viewing" photos enjoyable and informative for the user."""
def fetch_details(state: GraphState) -> GraphState:
    if state["image_path"] is not None:
        img = Image.open(state["image_path"])
        last_two_chats = get_last_two_chats(state["chat_history"])
        state["photo_description"] = model.generate_content([
            f"{Base_prompt} Refer to the image as view. The user's input is {state['user_input'][-1]}. Recent chat history: {last_two_chats}. Describe the view according to the user's request: ",
            img
        ]).text
        state["llm_output"] = state["photo_description"]
    else:
        state["llm_output"] = "Could not find the image"
    return state

def chat_bot(state: GraphState) -> GraphState:
    last_two_chats = get_last_two_chats(state["chat_history"])
    prompt=last_two_chats
    response = model.generate_content([f"You are an assistant for a blind user. You are extremely kind. You are unbelievably helpful and the Epitome of Innovation.Your tasks are strictly limited to taking a photo, answering users questions, take a video, saving taken video or photo, search the internet or be friends with the user. Recent chat history: {last_two_chats}. Answer the User's query: {state['user_input'][-1]}"])
    state["llm_output"] = response.text
    return state

def web_search(state: GraphState) -> GraphState:
    last_two_chats = get_last_two_chats(state["chat_history"])
    search_list = [f"Generate a short single line Web search query to answer user's query: {state['user_input'][-1]}. Recent chat history: {last_two_chats}. Use the image to understand the query if necessary"]
    if state["image_path"] is not None:
        img = Image.open(state["image_path"])
        search_list.append(img)
    search_prompt = model.generate_content(search_list)
    search_prompt = search_prompt.text.strip('"')
    search_results = google_search(api_key, cx, search_prompt)
    result = ""
    for item in search_results.get('items', [])[:3]:
        result += f"Title: {item['title']}\nSnippet: {item['snippet']}\n\n"
    if result == "":
        result = "No results found"
    state["feedback"] = result
    result = model.generate_content([f"You are an assistant for a blind user. You are very straightforward. Refer to the search result:{result}, and try to answer user's query: {state['user_input'][-1]}. Recent chat history: {last_two_chats}"])
    state["llm_output"] = result.text
    return state

def save_photo(state: GraphState) -> GraphState:
    state["llm_output"] = "Photo saved"
    return state

def take_video(state: GraphState) -> GraphState:
    state["llm_output"] = "Video taken"
    return state

def error(state: GraphState) -> GraphState:
    state["llm_output"] = "Some internal error occurred. Please retry"
    return state
def close_and_save(state:GraphState) -> GraphState:
    state["chat_history"].append({"UserMessage": state['user_input'][-1], "AssistantMessage": state['llm_output']})
    return state
workflow = StateGraph(GraphState)
workflow.add_node("first_user_input", first_user_input)

workflow.add_node("take_photo", take_photo)
workflow.add_node("fetch_details", fetch_details)
workflow.add_node("save_photo", save_photo)
workflow.add_node("take_video", take_video)
workflow.add_node("error", error)
workflow.add_node("web_search", web_search)
workflow.add_node("chat_bot", chat_bot)
workflow.add_node("close and save", close_and_save)
workflow.set_entry_point("first_user_input")
workflow.add_edge("take_photo", 'close and save')
workflow.add_edge("fetch_details",'close and save')
workflow.add_edge("save_photo", 'close and save')
workflow.add_edge("take_video", 'close and save')
workflow.add_edge('close and save',END)
workflow.add_conditional_edges(
    "first_user_input",
    decision,
    {
        "Take a photo": "take_photo",
        "Fetch details from the image": "fetch_details",
        "Save the photo": "save_photo",
        "Take a video": "take_video",
        "Error": "error",
        "Use Chat bot": "chat_bot",
        "Search the web for particular detail": "web_search",
    },
)

app = workflow.compile()
app = Flask(__name__)
compiled_workflow = workflow.compile()

@app.route('/invoke', methods=['POST'])
def invoke():
    data = request.form.to_dict()
    user_input = data.get("user_input")
    chat_history=data.get("chat_history")
    user_input = user_input.split(",") if user_input else ["describe_photo"]
    chat_history = chat_history if chat_history else []
    state = GraphState(
        user_input=user_input,
        photo_description=data.get('photo_description'),
        feedback=data.get('feedback'),
        llm_output=data.get('llm_output'),
        image_path=None,
        chat_history=chat_history
    )
    
    image_file = request.files.get('image')
    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_file.save(tmp_file.name)
            state['image_path'] = tmp_file.name
    
    result_state = compiled_workflow.invoke(state)
    
    if state.get('image_path'):
        os.remove(state['image_path'])
    
    return jsonify(result_state)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)