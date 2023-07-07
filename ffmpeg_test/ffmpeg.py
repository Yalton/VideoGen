
def ffmpegGen()

        temp_files = []

        print("generating video....")
        i, j = 0, 0
        while i < len(image_files) or j < len(tts_files):
            temp_file = f'temp{i+j}.mp4'
            temp_files.append(temp_file)

            # Prepare image stream
            if i < len(image_files):
                image_file = image_files[i]["filename"]
                image_duration = image_files[i]["duration"]
                video_stream = ffmpeg.input(image_file, loop=1, t=image_duration)
            else:
                video_stream = ffmpeg.input(temp_files[-1]).video  # Reuse the last image

            if j < len(tts_files):
                audio_file = tts_files[j]["filename"]
                audio_duration = tts_files[j]["duration"]
                silence_file = f'silence{j}.mp3'
                (
                    ffmpeg
                    .input('anullsrc=channel_layout=stereo:ar=44100', format='lavfi')
                    .output(silence_file, t=math.ceil(image_duration-audio_duration))
                    .run()
                )
                audio_stream = ffmpeg.concat(ffmpeg.input(audio_file), ffmpeg.input(silence_file), v=0, a=1)
            else:
                audio_stream = ffmpeg.input('anullsrc=channel_layout=stereo:ar=44100', format='lavfi', t=image_duration)



            # Combine the video and audio streams and output to the temporary file
            ffmpeg.output(video_stream, audio_stream, temp_file, vcodec='libx264', acodec='aac').run()

            if i < len(image_files):
                i += 1
            if j < len(tts_files):
                j += 1

        # Concatenate all the temporary video files
        inputs = []
        for i in temp_files:
            inputs.append(ffmpeg.input(i).video)
            inputs.append(ffmpeg.input(i).audio)
        ffmpeg.concat(*inputs, a=1, v=1).output(output_file).run()

        # Delete all the temporary video files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        return output_file

if __name__ == "__main__":