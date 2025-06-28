# livetranscribe with Kyutai Speech-To-Text on apple silicon

Works with English, French and German speech in real time on an m4 max MacBook Pro

## How to run

1. Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system
2.

```bash
uv run live_transcribe.py
```

### Experimental language support

by injecting short audio snippets in specific languages before starting to transcribe the microphone audio, we can force the model to try to use other languages than English or French. Currently works somewhat well with Spanish, German and Japanese:

```bash
uv run live_transcribe.py -l ger #German

uv run live_transcribe.py -l esp #Spanish

uv run live_transcribe.py -l jap #Japanese
```
