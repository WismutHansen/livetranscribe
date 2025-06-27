# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx",
#     "numpy",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
# ]
# ///

import argparse
import json
import queue
import sys
import signal

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils


def setup_signal_handler():
    """Set up graceful shutdown on Ctrl+C"""
    def signal_handler(signum, frame):
        print("\nStopping transcription...", file=sys.stderr)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)


class LiveTranscriber:
    def __init__(self, hf_repo: str, max_steps: int = 4096):
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
        self.audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)
        
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

    def audio_callback(self, indata, _frames, _time, _status):
        """Callback for audio input"""
        self.block_queue.put(indata.copy())

    def start_transcription(self):
        """Start live transcription from microphone"""
        print("Starting live transcription. Press Ctrl+C to stop.", file=sys.stderr)
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
                    other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                    other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                        :, :, :self.other_codebooks
                    ]
                    
                    # Generate text token
                    text_token = self.gen.step(other_audio_tokens[0])
                    text_token = text_token[0].item()
                    
                    # Process and output text
                    if text_token not in (0, 3):  # Skip padding and special tokens
                        text = self.text_tokenizer.id_to_piece(text_token)
                        text = text.replace("▁", " ")
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
        help="Hugging Face repository for the STT model"
    )
    parser.add_argument(
        "--max-steps", 
        type=int,
        default=4096,
        help="Maximum steps for the model"
    )
    parser.add_argument(
        "--list-devices", 
        action="store_true", 
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--device", 
        type=int, 
        help="Input device ID (use --list-devices to see options)"
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
        transcriber = LiveTranscriber(args.hf_repo, args.max_steps)
        transcriber.start_transcription()
    except KeyboardInterrupt:
        print("\nTranscription stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()