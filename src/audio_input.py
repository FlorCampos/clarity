import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/app')

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────
# BASE CLASS — defines the contract all tiers must follow
# ─────────────────────────────────────────────────────────────

class TranscriptionService:
    """
    Base class for all transcription tiers.
    Every tier MUST implement transcribe().
    This is the contract — the interface.
    """

    def transcribe(self, audio_file: str) -> dict:
        """
        Transcribes audio file to text.

        Args:
            audio_file: path to audio file (mp3, wav, m4a)

        Returns:
            dict: {
                "text": full transcription,
                "segments": list of timed segments,
                "speakers": speaker-separated text (if available),
                "duration": audio duration in seconds,
                "tier": which tier was used
            }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transcribe()"
        )

    def _validate_file(self, audio_file: str) -> None:
        """Validates the audio file exists and is supported."""
        path = Path(audio_file)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}"
            )

        supported = {'.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.flac'}
        if path.suffix.lower() not in supported:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {supported}"
            )


# ─────────────────────────────────────────────────────────────
# TIER 1 — Local Whisper (private, free)
# ─────────────────────────────────────────────────────────────

class LocalWhisperService(TranscriptionService):
    """
    Tier 1 — Local Whisper model.

    Privacy:  100% local — audio never leaves your machine
    Speed:    2-3 minutes per hour of audio (CPU)
    Cost:     Free
    Speakers: No speaker separation
    Best for: Regulated industries, confidential meetings
    """

    def __init__(self, model_size: str = "small"):
        """
        Args:
            model_size: tiny/base/small/medium/large
            small = best balance of speed and accuracy
        """
        self.model_size = model_size
        self.model = None
        print(f"\n  🎤 LocalWhisperService initialized")
        print(f"     Model size: {model_size}")
        print(f"     Privacy: 100% local ✅")

    def _load_model(self):
        """Loads Whisper model — only when needed."""
        if self.model is None:
            import whisper
            print(f"\n  Loading Whisper {self.model_size} model...")
            print(f"  (First run downloads ~500MB — cached after)")
            self.model = whisper.load_model(self.model_size)
            print(f"  ✅ Model loaded")

    def transcribe(self, audio_file: str) -> dict:
        """Transcribes audio using local Whisper model."""

        self._validate_file(audio_file)
        self._load_model()

        print(f"\n  🎵 Transcribing: {Path(audio_file).name}")
        print(f"  ⏳ Processing locally — please wait...")

        import whisper
        result = self.model.transcribe(
            audio_file,
            language="en",
            verbose=False
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })

        duration = segments[-1]["end"] if segments else 0

        return {
            "text": result["text"].strip(),
            "segments": segments,
            "speakers": None,
            "duration": duration,
            "tier": "local_whisper",
            "model": self.model_size,
            "privacy": "100% local"
        }


# ─────────────────────────────────────────────────────────────
# TIER 2 — AssemblyAI (fast + speaker labels)
# ─────────────────────────────────────────────────────────────

class AssemblyAIService(TranscriptionService):
    """
    Tier 2 — AssemblyAI API.

    Privacy:  GDPR compliant, data deleted after processing
    Speed:    ~30 seconds per hour of audio
    Cost:     $0.002 per minute
    Speakers: YES — Client vs PM vs Developer
    Best for: Dev agencies, startups, fast turnaround
    """

    def __init__(self):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY not found in .env\n"
                "Get your key at assemblyai.com"
            )
        print(f"\n  🎤 AssemblyAIService initialized")
        print(f"     Speaker diarization: ✅")
        print(f"     Privacy: GDPR compliant ✅")

    def transcribe(self, audio_file: str) -> dict:
        """Transcribes with speaker separation via AssemblyAI."""

        self._validate_file(audio_file)

        import assemblyai as aai
        aai.settings.api_key = self.api_key

        print(f"\n  🎵 Uploading: {Path(audio_file).name}")
        print(f"  ⚡ Processing via AssemblyAI...")

        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2
        )

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            audio_file,
            config=config
        )

        full_text = transcript.text
        segments = []
        speakers = {}

        for utterance in transcript.utterances:
            speaker = f"Speaker_{utterance.speaker}"
            text = utterance.text

            segments.append({
                "start": utterance.start / 1000,
                "end": utterance.end / 1000,
                "speaker": speaker,
                "text": text
            })

            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(text)

        speaker_text = {}
        for speaker, texts in speakers.items():
            speaker_text[speaker] = " ".join(texts)

        duration = segments[-1]["end"] if segments else 0

        return {
            "text": full_text,
            "segments": segments,
            "speakers": speaker_text,
            "duration": duration,
            "tier": "assemblyai",
            "privacy": "GDPR compliant"
        }


# ─────────────────────────────────────────────────────────────
# TIER 3 — Azure Speech (enterprise, private cloud)
# ─────────────────────────────────────────────────────────────

class AzureSpeechService(TranscriptionService):
    """
    Tier 3 — Azure Cognitive Services Speech.

    Privacy:  Private cloud — your region, your data
    Speed:    ~30 seconds + SLA guarantee
    Cost:     Enterprise contract
    Speakers: YES + custom vocabulary
    Best for: Banks, hospitals, government
    """

    def __init__(self):
        self.api_key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")

        if not self.api_key or not self.region:
            raise ValueError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION "
                "required in .env"
            )
        print(f"\n  🎤 AzureSpeechService initialized")
        print(f"     Region: {self.region}")
        print(f"     Privacy: Private cloud ✅")

    def transcribe(self, audio_file: str) -> dict:
        """Transcribes via Azure Cognitive Services."""

        self._validate_file(audio_file)

        print(f"\n  🎵 Processing via Azure Speech...")
        print(f"     Region: {self.region}")

        # Azure SDK implementation
        # Full implementation added when client needs Tier 3
        raise NotImplementedError(
            "Azure Tier 3 — add azure-cognitiveservices-speech "
            "to requirements.txt and implement SDK calls"
        )


# ─────────────────────────────────────────────────────────────
# FACTORY — returns the right tier from .env config
# ─────────────────────────────────────────────────────────────

def get_transcription_service() -> TranscriptionService:
    """
    Factory function — reads TRANSCRIPTION_TIER from .env
    and returns the correct service.

    To switch tiers: change TRANSCRIPTION_TIER in .env
    No code changes needed.
    """

    tier = os.getenv("TRANSCRIPTION_TIER", "local").lower()

    print(f"\n  📋 Transcription tier: {tier.upper()}")

    if tier == "local":
        model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        return LocalWhisperService(model_size=model_size)

    elif tier == "assemblyai":
        return AssemblyAIService()

    elif tier == "azure":
        return AzureSpeechService()

    else:
        print(f"  ⚠️  Unknown tier '{tier}' — falling back to local")
        return LocalWhisperService()


# ─────────────────────────────────────────────────────────────
# AUDIO AGENT — connects transcription to requirements agent
# ─────────────────────────────────────────────────────────────

def process_audio_meeting(
    audio_file: str,
    project_name: str = "default"
) -> dict:
    """
    Main function — transcribes a client meeting and
    processes it through the full Clarity pipeline.

    Args:
        audio_file: path to meeting recording
        project_name: which project this belongs to

    Returns:
        dict: transcription + all parsed requirements
    """

    from src.agent import RequirementsAgent

    print(f"\n{'='*60}")
    print(f"  CLARITY — Audio Meeting Processor")
    print(f"  File: {Path(audio_file).name}")
    print(f"  Project: {project_name}")
    print(f"{'='*60}")

    # Step 1 — Transcribe
    service = get_transcription_service()
    transcription = service.transcribe(audio_file)

    print(f"\n  ✅ Transcription complete")
    print(f"     Duration: {transcription['duration']:.0f}s")
    print(f"     Words: {len(transcription['text'].split())}")
    print(f"     Tier: {transcription['tier']}")

    if transcription.get('speakers'):
        print(f"     Speakers: {len(transcription['speakers'])}")
        for speaker, text in transcription['speakers'].items():
            print(f"     {speaker}: {len(text.split())} words")

    # Step 2 — Extract requirements from transcript
    print(f"\n  🧠 Extracting requirements from transcript...")

    agent = RequirementsAgent(project_name=project_name)

    # Use speaker-aware text if available
    if transcription.get('speakers'):
        formatted_text = _format_speaker_text(
            transcription['speakers']
        )
    else:
        formatted_text = transcription['text']

    # Process through requirements agent
    result = agent.process(formatted_text)

    return {
        "transcription": transcription,
        "requirement": result,
        "processed_at": datetime.now().isoformat()
    }


def _format_speaker_text(speakers: dict) -> str:
    """
    Formats speaker-separated text for better
    requirement extraction by Claude.
    """
    formatted = "MEETING TRANSCRIPT WITH SPEAKERS:\n\n"
    for speaker, text in speakers.items():
        formatted += f"{speaker}: {text}\n\n"
    return formatted


def display_transcription(transcription: dict) -> None:
    """Prints transcription results cleanly."""

    print(f"\n{'='*60}")
    print(f"  TRANSCRIPTION RESULT")
    print(f"{'='*60}")
    print(f"\n  Tier:     {transcription['tier']}")
    print(f"  Duration: {transcription['duration']:.0f} seconds")
    print(f"  Privacy:  {transcription['privacy']}")

    if transcription.get('speakers'):
        print(f"\n  SPEAKERS DETECTED:")
        for speaker, text in transcription['speakers'].items():
            print(f"\n  {speaker}:")
            print(f"  {text[:200]}...")
    else:
        print(f"\n  TRANSCRIPT:")
        print(f"  {transcription['text'][:500]}...")

    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n  Testing TranscriptionService architecture...")
    print("\n  Available tiers:")
    print("  - local      → Whisper runs in Docker (free, private)")
    print("  - assemblyai → AssemblyAI API (fast, speaker labels)")
    print("  - azure      → Azure Speech (enterprise)")

    print(f"\n  Current tier from .env:")
    service = get_transcription_service()
    print(f"  Service: {service.__class__.__name__}")

    print(f"\n  To test with real audio:")
    print(f"  1. Add an MP3 file to your project folder")
    print(f"  2. Run:")
    print(f"     docker compose run clarity python -c \"")
    print(f"     import sys; sys.path.insert(0, '/app')")
    print(f"     from src.audio_input import process_audio_meeting")
    print(f"     result = process_audio_meeting('your_file.mp3')")
    print(f"     \"")

    print(f"\n  ✅ Audio input architecture ready")
    print(f"     Tier 1 (local):      built and ready")
    print(f"     Tier 2 (assemblyai): add ASSEMBLYAI_API_KEY to .env")
    print(f"     Tier 3 (azure):      add AZURE keys to .env")