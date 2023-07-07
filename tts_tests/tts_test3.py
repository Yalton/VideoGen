from TTS.api import TTS

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one

i = 0
while i < 65: 
    model_name = TTS.list_models()[i]
    # Init TTS
    tts = TTS(model_name)
    if tts.speakers: 
        tts.tts_to_file(text="This is a test! This is also a test!!", speaker=tts.speakers[0], language="en", file_path=f"output{i}.wav")
    else: 
        tts.tts_to_file(text="This is a test! This is also a test!!", language="en", file_path=f"output{i}.wav")

    i+=1
