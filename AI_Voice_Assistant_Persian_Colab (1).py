# ============================================================
# 🎙️ AI VOICE ASSISTANT - FREE / NO API KEY
# Google Colab
#
# Microphone -> Whisper -> Local AI -> Persian TTS -> Audio
# ============================================================

!pip -q install -U openai-whisper transformers accelerate sentencepiece soundfile scipy
!apt-get -qq update
!apt-get -qq install -y ffmpeg

import torch
import whisper
import base64

from transformers import pipeline, VitsModel, AutoTokenizer
from IPython.display import Audio, display
from google.colab import output


# ============================================================
# 1. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("===================================")
print("Device:", device)
print("===================================")

if device == "cuda":
    print("✅ GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU is not active.")
    print("Colab: Runtime -> Change runtime type -> T4 GPU")


# ============================================================
# 2. WHISPER - SPEECH TO TEXT
# ============================================================

print("\n🎤 Loading Whisper...")

whisper_model = whisper.load_model("base")

print("✅ Whisper loaded")


# ============================================================
# 3. LOCAL AI
# ============================================================

print("\n🤖 Loading local AI...")

ai_device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=ai_device
)

print("✅ Local AI loaded")


# ============================================================
# 4. PERSIAN TTS
# ============================================================

print("\n🔊 Loading Persian TTS...")

TTS_MODEL = "facebook/mms-tts-fas"

tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
tts_model = VitsModel.from_pretrained(TTS_MODEL)
tts_model = tts_model.to(device)

print("✅ Persian TTS loaded")


# ============================================================
# 5. RECORD MICROPHONE
# ============================================================

def record_audio(seconds=6):

    print(f"\n🎙️ Recording for {seconds} seconds...")

    js = f"""
    async function recordAudio() {{

        const stream =
            await navigator.mediaDevices.getUserMedia({{audio: true}});

        const recorder = new MediaRecorder(stream);
        const chunks = [];

        recorder.ondataavailable =
            event => chunks.push(event.data);

        recorder.start();

        await new Promise(
            resolve => setTimeout(resolve, {seconds * 1000})
        );

        recorder.stop();

        await new Promise(
            resolve => recorder.onstop = resolve
        );

        stream.getTracks().forEach(
            track => track.stop()
        );

        const blob =
            new Blob(chunks, {{type: 'audio/webm'}});

        const buffer =
            await blob.arrayBuffer();

        const bytes =
            new Uint8Array(buffer);

        let binary = "";
        const chunkSize = 0x8000;

        for (
            let i = 0;
            i < bytes.length;
            i += chunkSize
        ) {{
            binary += String.fromCharCode(
                ...bytes.subarray(i, i + chunkSize)
            );
        }}

        return btoa(binary);
    }}

    recordAudio();
    """

    data = output.eval_js(js)
    audio_bytes = base64.b64decode(data)

    path = "/content/user_audio.webm"

    with open(path, "wb") as f:
        f.write(audio_bytes)

    print("✅ Recording finished")

    return path


# ============================================================
# 6. SPEECH -> TEXT
# ============================================================

def speech_to_text(audio_file):

    print("\n📝 Speech -> Text...")

    result = whisper_model.transcribe(
        audio_file,
        language="fa"
    )

    return result["text"].strip()


# ============================================================
# 7. AI RESPONSE
# ============================================================

conversation = [
    {
        "role": "system",
        "content": (
            "تو یک دستیار صوتی فارسی هستی. "
            "کوتاه، طبیعی و واضح جواب بده. "
            "پاسخ‌ها را به زبان فارسی بده."
        )
    }
]


def ask_ai(user_text):

    conversation.append({
        "role": "user",
        "content": user_text
    })

    prompt = ""

    for message in conversation:
        prompt += (
            message["role"]
            + ": "
            + message["content"]
            + "\n"
        )

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )

    answer = result[0]["generated_text"].strip()

    conversation.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ============================================================
# 8. TEXT -> PERSIAN SPEECH
# ============================================================

def text_to_speech(
    text,
    output_file="/content/assistant.wav"
):

    print("\n🔊 Text -> Speech...")

    inputs = tts_tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_audio = tts_model(**inputs).waveform

    audio_numpy = (
        output_audio
        .cpu()
        .numpy()
        .squeeze()
    )

    sample_rate = tts_model.config.sampling_rate

    import scipy.io.wavfile as wavfile

    wavfile.write(
        output_file,
        sample_rate,
        audio_numpy
    )

    return output_file


# ============================================================
# 9. COMPLETE VOICE ASSISTANT
# ============================================================

def voice_assistant(seconds=6):

    print("\n===================================")
    print("🎙️ AI VOICE ASSISTANT")
    print("===================================")

    # Microphone
    audio_file = record_audio(seconds)

    # Speech -> Text
    user_text = speech_to_text(audio_file)

    print("\n👤 شما:")
    print(user_text)

    # AI
    print("\n🤖 AI در حال فکر کردن...")

    answer = ask_ai(user_text)

    print("\n🤖 دستیار:")
    print(answer)

    # Text -> Speech
    output_file = text_to_speech(answer)

    # Play
    print("\n🎧 Playing...")

    display(
        Audio(
            output_file,
            autoplay=True
        )
    )

    return {
        "user_text": user_text,
        "answer": answer,
        "audio": output_file
    }


# ============================================================
# 10. RUN
# ============================================================

result = voice_assistant(seconds=6)
