#from langchain.chains import Chain
from pydantic import BaseModel, Field, root_validator, validator

# Import Langchain Tools 
from langchain.chains.base import Chain
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import load_tools
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.experimental import AutoGPT
from langchain.tools.human.tool import HumanInputRun

from langchain.utilities import SearxSearchWrapper
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Import Misc Libraries 
import os
import io
import re
import csv
import math
import glob
import faiss
import ffmpeg
import base64
import torch
import torchaudio
import requests
import subprocess
import configparser
from TTS.api import TTS
from pydub import AudioSegment
from PIL import Image, PngImagePlugin
from audiocraft.models import MusicGen
from typing import Optional, Type, List
from audiocraft.data.audio import audio_write

def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

def log_output_file(output_file_path):
    file_extension = os.path.splitext(output_file_path)[-1].lower()  # Get file extension

    # Map common file extensions to file types
    file_type_mapping = {
        '.png': 'Image',
        '.mp4': 'Video',
        '.mp3': 'TTS_Audio',
        '.wav': 'Music',
    }

    # Get file type from mapping, or use 'Unknown' if extension is not in the mapping
    file_type = file_type_mapping.get(file_extension, 'Unknown')

    with open("medialog.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file_type, output_file_path])

def clear_log():
    open("medialog.csv", 'w').close()

clear_log()

ROOT_DIR = "./data/"

config = read_config()

def generate_silence_file(filename: str = 'silence.mp3', duration: int = 10000):
    """Generate a silence audio file.

    Parameters:
        filename (str): The filename of the silence audio file. Defaults to 'silence.mp3'.
        duration (int): The duration of the silence in milliseconds. Defaults to 10000 (10 seconds).
    """
    silence = AudioSegment.silent(duration=duration)
    silence.export(filename, format="mp3")

@tool
def check_files():
    """Check files which have been generated previously; whenever a file is generated it is logged to a csv, thus this tool can help verify which files exist, and which do not"""
    files = []
    with open("medialog.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_path = row[0]
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                print(f"File {file_path} does not exist")
    return files

class FileCheckTool(BaseTool):
    name = "file_check"
    description = "Gives you a list of all media files which have been generated thus far; useful for validating which media to use in the video generation step"
    #args_schema: Type[FileCheckSchema] = FileCheckSchema

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        files = []
        with open("medialog.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                file_type = row[0]
                file_path = row[1]
                if os.path.exists(file_path):
                    print(f"{file_type}: {file_path}")
                    files.append(file_path)
                else:
                    print(f"File {file_path} does not exist")
        return files

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class ImageSchema(BaseModel):
    prompt: str = Field(description="Prompt for video generator; should be a text description of what the final image should be. Input should be a simple string")
    output_File: str = Field(default=None, description="Name of file where generated image will end up; Input should be a file name with the extension of .png")
    @validator('output_File', pre=True, always=True)
    def generate_default_output_file(cls, v, values) -> str:
        if v is None and 'prompt' in values:
            # Remove non-alphanumeric characters and spaces from the prompt
            prompt_as_filename = re.sub('[^0-9a-zA-Z]+', '_', values['prompt'])
            # Trim the filename if it's too long and append .png
            v = f"{prompt_as_filename[:50]}.png"
        return v

class ImageGenerator(BaseTool):
    name = "generate_image"
    description = "useful for when you need to generate an image using stable diffusion; the input to this tool is a prompt for the image generator to create, and the output file for the image; both are required"
    args_schema: Type[ImageSchema] = ImageSchema

    def _run(
        self, prompt: str, output_File: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        #url = "http://192.168.50.245:7860"

        if(output_File == ""): 
            return "ERROR: Output File empty, Output file required to be non empty"

        url = config.get("ENV_VARS", "StableDiffusionWebUI")
        
        altered_prompt = f"(((masterpiece))),best quality, intricate, highly detailed, textile shading, ((cinematic lighting)), {prompt}"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry, artist name, nsfw, lowres, bad anatomy, bad hands, text error, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, standard quality, jpeg artifacts, signature, watermark, username, blurry"

        payload = {
            "prompt": altered_prompt,
            "negative_prompt": negative_prompt,
            "width": 960, 
            "height": 540,
            "steps": 26,
            "cfg_scale": 7,
            "restore_faces": "true",
            "sd_model_checkpoint": "icbinpICantBelieveIts_v8.safetensors",
            "sampler_index": "Euler a",
            "filter_nsfw": "true",
            "CLIP_stop_at_last_layers": 2
        }

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

        print(response)

        r = response.json()

        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

            png_payload = {
                "image": "data:image/png;base64," + i
            }
            response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", response2.json().get("info"))
            image.save(output_File, pnginfo=pnginfo)

        log_output_file(output_File)
        return output_File

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class TextToSpeechSchema(BaseModel):
    TextToSpeak: str = Field(description="Text to be spoken by the tts model")
    outputFile: str = Field(default=None, description="File where output speech should end up")
    @validator('outputFile', pre=True, always=True)
    def generate_default_output_file(cls, v, values) -> str:
        if v is None and 'prompt' in values:
            # Remove non-alphanumeric characters and spaces from the prompt
            prompt_as_filename = re.sub('[^0-9a-zA-Z]+', '_', values['prompt'])
            # Trim the filename if it's too long and append .png
            v = f"{prompt_as_filename[:50]}.mp3"
        return v

class TextToSpeechTool(BaseTool):
    name = "text2_speech"
    description = "useful for when you need to convert text to speech and save it to an output file. Both the text to convert and the name of the desired output file are required"
    args_schema: Type[TextToSpeechSchema] = TextToSpeechSchema

    def _run(
        self, TextToSpeak: str, outputFile: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if(outputFile == ""): 
            return "ERROR: Output File empty, Output file required to be non empty"

        model_name = TTS.list_models()[0]
        # Init TTS
        tts = TTS(model_name)
        # Text to speech to a file
        tts.tts_to_file(text=TextToSpeak, speaker=tts.speakers[0], language=tts.languages[0], file_path=outputFile)
        log_output_file(outputFile)
        return outputFile

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class MusicGenerationSchema(BaseModel):
    musicDescription: str = Field(description="type of music to be generated by the tool, accepts genres and descriptions; description should go before genre")
    outputFile: str = Field(default=None, description="File where output music file should end up, no file extension required .wav will be appended to name provided; music sample will always be 10 seconds long")
    @validator('outputFile', pre=True, always=True)
    def generate_default_output_file(cls, v, values) -> str:
        if v is None and 'prompt' in values:
            # Remove non-alphanumeric characters and spaces from the prompt
            prompt_as_filename = re.sub('[^0-9a-zA-Z]+', '_', values['prompt'])
            # Trim the filename if it's too long and append .png
            v = f"{prompt_as_filename[:50]}"
        return v

class MusicGenerationTool(BaseTool):
    name = "music_generator"
    description = "useful for when you need to convert text to music and save it to an output file"
    args_schema: Type[MusicGenerationSchema] = MusicGenerationSchema

    def _run(
        self, musicDescription: str, outputFile: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        torch.cuda.empty_cache()


        # If outputFile ends with .wav, remove it
        if outputFile.endswith('.wav'):
            outputFile = outputFile[:-4]

        model = MusicGen.get_pretrained('small')
        model.set_generation_params(duration=10)  # generate 10 seconds.

        description = []
        description.append(musicDescription)
        wav = model.generate(description)  

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{outputFile}', one_wav.cpu(), model.sample_rate, strategy="loudness")
        log_output_file(f"{outputFile}.wav")
        return f"{outputFile}.wav"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class SearchSchema(BaseModel):
    query: str = Field(description="should be a search query")
    num_results: int =  Field(description="Number of results desired for engine to retrieve")
    category: str =  Field(description="Category to search on: Supported types are General, Files, Images, It, Map, Music, News, Science, Shopping, Social, Media, Videos, Web")

    
    # engine: str = Field(description="should be a search engine")
    # gl: str = Field(description="should be a country code")
    # hl: str = Field(description="should be a language code")

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"
    args_schema: Type[SearchSchema] = SearchSchema

    def _run(
        self,
        query: str,
        num_results: int,
        category: str, 
        # engine: str = "google",
        # gl: str = "us",
        # hl: str = "en",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        search = SearxSearchWrapper(searx_host="https://searx.billbert.co/")

        results = search.results(query, num_results=num_results, categories=category, time_range='year')

        #search_wrapper = SerpAPIWrapper(params={"engine": engine, "gl": gl, "hl": hl})
        return results

    async def _arun(
        self,
        query: str,
        engine: str = "google",
        gl: str = "us",
        hl: str = "en",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class File(BaseModel):
    filename: str = Field(description="The filename of the image or audio file.")
    duration: int = Field(description="The duration for which the file should be played or displayed.")

class CreateVideoSchema(BaseModel):
    image_files: List[File] = Field(description="This argument is required not to be empty. List of image files to be edited together. Dictionary accepts arguments of filename and duration. Duration of each image_file present in the output should also be specified. Files need to exist in directory.")
    tts_files: List[File] = Field(description="This argument is required not to be empty. List of mp3 pre-generated audio files to be edited together in a sequence. Dictionary accepts arguments of filename and duration.  Duration of each tts_file present in the output should also be specified. Files need to exist in directory.")
    background_music: List[File] = Field(description="This argument is required not to be empty. Background music to be played throughout the video. Accepts filename and duration. File needs to exist in directory.")
    output_file: str = Field(description="Destination where final video product should end up.")
    framerate: int = Field(description="Framerate of final video product.")

class CreateVideoTool(BaseTool):
    name = "create_video"
    description = "Image files are REQUIRED; TTS Files are REQUIRED; Useful for when you need to edit audio, video, or image files together into a video. All input files MUST EXIST prior to running this tool. Tool should only be used after image and audio files have been generated"
    args_schema: Type[CreateVideoSchema] = CreateVideoSchema

    def _run(
        self, image_files: List[File], tts_files: List[File], background_music: List[File], output_file: str, framerate: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Create a video from a list of image files and an audio file.

        Parameters:
            image_files (list): A list of paths to image files.
            audio_file (str): The path to the audio file.
            output_file (str): The path where the output video file will be saved.
            framerate (int, optional): The number of frames per second in the video. Defaults to 1.
        """
        
        if len(image_files) == 0 or len(tts_files) == 0:
            return  "Empty list of files passed in; function requires Image, TTS_Files, and Background Music"
        # Check if all files exist
        non_existent_files = []

        # Check for image files
        for image_file in image_files:
            if not os.path.isfile(image_file["filename"]):
                non_existent_files.append(image_file["filename"])

        # Check for tts files
        for tts_file in tts_files:
            if not os.path.isfile(tts_file["filename"]):
                non_existent_files.append(tts_file["filename"])

        # Check for background music files
        for bgm_file in background_music:
            if not os.path.isfile(bgm_file["filename"]):
                non_existent_files.append(bgm_file["filename"])

        if len(non_existent_files) > 0:
            non_existent_files.insert(0, "VIDEO CREATION FAILED; following files do not exist:")
            return non_existent_files

        # Calculate total audio duration and divide by the number of images
        temp_files = []
        total_audio_duration = sum([tts_file["duration"] for tts_file in tts_files]) 
        image_duration = total_audio_duration / len(image_files) 

        print("generating video....")
        i, j = 0, 0

        for i in range(len(image_files)):
            temp_file = f'temp{i}.mp4'
            temp_files.append(temp_file)

            image_file = image_files[i]["filename"]
            video_stream = ffmpeg.input(image_file, loop=1, t=image_duration)

            ffmpeg.output(video_stream, temp_file, vcodec='libx264').run()

        # Concatenate all the temporary video files
        video_files = [ffmpeg.input(temp_file).video for temp_file in temp_files]
        video_stream = ffmpeg.concat(*video_files, v=1)
        
        # Get audio file and background music
        audio_file = tts_files[0]["filename"]
        audio_stream = ffmpeg.input(audio_file).audio
        audio_stream = ffmpeg.filter_(audio_stream, 'adelay', '1s')
        
        if background_music and len(background_music) > 0:
            bgm_filename = background_music[0]["filename"]
            if os.path.isfile(bgm_filename):
                bgm_stream = ffmpeg.input(bgm_filename).audio
                bgm_stream = ffmpeg.filter_(bgm_stream, 'volume', 0.15)
                audio_stream = ffmpeg.filter([audio_stream, bgm_stream], 'amix')


        ffmpeg.output(video_stream, audio_stream, output_file, acodec='aac').run()

        # Delete all the temporary video files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        log_output_file(output_file)
        return output_file


    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

llm = ChatOpenAI(temperature=0.1, openai_api_key=config.get("API_KEYS", "OpenAI"))

generate_silence_file()

image_generator = ImageGenerator()
text2_speech = TextToSpeechTool()
create_video = CreateVideoTool()
search_web = CustomSearchTool()
file_check = FileCheckTool()
music_gen = MusicGenerationTool()
human_input_tool = HumanInputRun()

tools = [
    image_generator,
    text2_speech,
    create_video,
    search_web,
    file_check,
    music_gen,
    human_input_tool
]

planner = load_chat_planner(llm)

executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

human_input = input("Topic to generate a video on > ")


topic = "Generate me a video on the topic of {human_input}"
agent.run(
    f"VideoGen is an agent with tools allowing it to generate videos about any topic; follow steps laid out in order, do not deviate, do not stop your task without generating a video \
    __Respect the schemas of all provided tools; and provide values for all inputs in the schema__ \
    __Step 1: Use the search_web tool to gather up to date information about {human_input}.__  \
    __Step 2: Use the text2_speech tool to narrate the information found and translate it into audio files; format the information in an informative manner, and do not mention that it cane from search results. ensure an output file is provided to this tool__ \
    __Step 3: Generate at least 5 images which are pertinent to the information discussed; and the information found online. When generating images; ensure each file has a unique name so they do not overwrite eachother.__ \
    __Step 4: Generate background music using the music_gen tool, with a genere that best fits the information discussed.__  \
    __Step 5: IMPORTANT: Get a list of images and audio using file_check tool; this list should be used in Step 6 to generate the video.__ \
    __Step 6: Create a video with the create_video tool;  use the list retrieved in Step 5 the create_video tool will fail if file that does not exist are passed as arguments. \
    Video To generate: {topic}"
)

