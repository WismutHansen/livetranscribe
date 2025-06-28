# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx",
#     "numpy",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
#     "soundfile",
# ]
# ///

import argparse
import json
import os
import queue
import sys
import signal

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
import soundfile as sf
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils


def setup_signal_handler():
    """Set up graceful shutdown on Ctrl+C"""

    def signal_handler(signum, frame):
        print("\nStopping transcription...", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


class LiveTranscriber:
    def __init__(self, hf_repo: str, max_steps: int = 4096, lang_ref: str = None):
        self.max_steps = max_steps
        self.block_queue = queue.Queue()

        # Load model configuration
        lm_config = hf_hub_download(hf_repo, "config.json")
        with open(lm_config, "r") as fobj:
            lm_config = json.load(fobj)

        # Download model files
        mimi_weights = hf_hub_download(hf_repo, lm_config["mimi_name"])
        moshi_name = lm_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(hf_repo, moshi_name)
        tokenizer = hf_hub_download(hf_repo, lm_config["tokenizer_name"])

        # Initialize model
        lm_config = models.LmConfig.from_config_dict(lm_config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        # Apply quantization if needed
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)

        print("Loading model weights...", file=sys.stderr)
        model.load_weights(moshi_weights, strict=True)

        print("Loading text tokenizer...", file=sys.stderr)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

        print("Loading audio tokenizer...", file=sys.stderr)
        generated_codebooks = lm_config.generated_codebooks
        other_codebooks = lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        self.audio_tokenizer = rustymimi.Tokenizer(
            mimi_weights, num_codebooks=mimi_codebooks
        )

        print("Warming up model...", file=sys.stderr)
        model.warmup()

        self.gen = models.LmGen(
            model=model,
            max_steps=self.max_steps,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )
        self.other_codebooks = other_codebooks
        self.lang_ref = lang_ref
        self.ref_audio_processed = False
        self.byte_buffer = []  # Buffer for accumulating byte tokens

    def audio_callback(self, indata, _frames, _time, _status):
        """Callback for audio input"""
        self.block_queue.put(indata.copy())

    def _load_and_process_ref_audio(self):
        """Load and process reference audio file"""
        if not self.lang_ref:
            return

        ref_audio_path = os.path.join("ref_audio", f"{self.lang_ref}.mp3")
        if not os.path.exists(ref_audio_path):
            print(
                f"Warning: Reference audio file {ref_audio_path} not found",
                file=sys.stderr,
            )
            return

        try:
            # Check if we need to resample the audio
            temp_path = None
            audio_info = sf.info(ref_audio_path)
            
            if audio_info.samplerate != 24000:
                import tempfile
                import subprocess
                
                # Create temporary resampled file
                temp_path = tempfile.mktemp(suffix='.wav')
                subprocess.run([
                    'ffmpeg', '-i', ref_audio_path, 
                    '-ar', '24000', '-ac', '1', 
                    '-y', temp_path
                ], check=True, capture_output=True)
                audio_path = temp_path
            else:
                audio_path = ref_audio_path
            
            # Load audio file
            audio_data, sample_rate = sf.read(audio_path)
            # Convert to numpy array first, then to MLX
            import numpy as np
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            audio_data = audio_data.astype(np.float32)
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

            # Process in chunks similar to live audio
            chunk_size = 1920
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad the last chunk
                    import numpy as np
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # Match the live audio processing format exactly
                block = chunk[None, :, 0] if len(chunk.shape) > 1 else chunk[None, :]

                # Encode audio exactly like live processing
                other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, : self.other_codebooks
                ]

                # Generate text token (priming the model)
                text_token = self.gen.step(other_audio_tokens[0])

            print(f"Processed reference audio: {self.lang_ref}.mp3", file=sys.stderr)

        except Exception as e:
            print(f"Error processing reference audio: {e}", file=sys.stderr)

    def start_transcription(self):
        """Start live transcription from microphone"""
        print("Starting live transcription. Press Ctrl+C to stop.", file=sys.stderr)

        # Process reference audio if provided
        if self.lang_ref:
            print(f"Loading language reference: {self.lang_ref}", file=sys.stderr)
            self._load_and_process_ref_audio()

        print("Transcription output:", file=sys.stderr)
        print("-" * 50, file=sys.stderr)

        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=24000,
            blocksize=1920,
            callback=self.audio_callback,
        ):
            while True:
                try:
                    block = self.block_queue.get()
                    block = block[None, :, 0]

                    # Encode audio
                    other_audio_tokens = self.audio_tokenizer.encode_step(
                        block[None, 0:1]
                    )
                    other_audio_tokens = mx.array(other_audio_tokens).transpose(
                        0, 2, 1
                    )[:, :, : self.other_codebooks]

                    # Generate text token
                    text_token = self.gen.step(other_audio_tokens[0])
                    text_token = text_token[0].item()

                    # Process and output text
                    if text_token not in (0, 3):  # Skip padding and special tokens
                        text = self.text_tokenizer.id_to_piece(text_token)
                        text = text.replace("▁", " ")
                        
                        # Handle byte sequence tokens
                        import re
                        if re.match(r'^<0x[0-9A-Fa-f]{2}>$', text.strip()):
                            # This is a byte token, add to buffer
                            hex_value = re.search(r'<0x([0-9A-Fa-f]{2})>', text).group(1)
                            self.byte_buffer.append(int(hex_value, 16))
                            # Try to decode accumulated bytes
                            try:
                                decoded = bytes(self.byte_buffer).decode('utf-8')
                                # Successfully decoded, output and clear buffer
                                print(decoded, end="", flush=True)
                                self.byte_buffer = []
                            except UnicodeDecodeError:
                                # Need more bytes, continue accumulating
                                pass
                        else:
                            # Regular text token
                            # First flush any pending bytes as-is if they can't be decoded
                            if self.byte_buffer:
                                try:
                                    decoded = bytes(self.byte_buffer).decode('utf-8', errors='ignore')
                                    if decoded:
                                        print(decoded, end="", flush=True)
                                except:
                                    pass
                                self.byte_buffer = []
                            print(text, end="", flush=True)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error processing audio: {e}", file=sys.stderr)
                    continue


def main():
    parser = argparse.ArgumentParser(description="Live speech-to-text transcription")
    parser.add_argument(
        "--hf-repo",
        default="kyutai/stt-1b-en_fr-mlx",
        help="Hugging Face repository for the STT model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=4096, help="Maximum steps for the model"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
    parser.add_argument(
        "--lang",
        "-l",
        choices=["esp", "ger", "jap"],
        help="Language for reference audio injection (esp=Spanish, ger=German, jap=Japanese)",
    )

    args = parser.parse_args()

    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return

    if args.device is not None:
        sd.default.device[0] = args.device

    setup_signal_handler()

    try:
        transcriber = LiveTranscriber(args.hf_repo, args.max_steps, args.lang)
        transcriber.start_transcription()
    except KeyboardInterrupt:
        print("\nTranscription stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

