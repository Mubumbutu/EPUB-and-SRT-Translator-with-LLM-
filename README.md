![aplikacja](https://github.com/user-attachments/assets/b5e0af62-8eda-47f3-8b1f-ae4ae8fc89ce)

I needed a tool that could harness large language models to translate both e-books in `.epub` format and movie subtitles in `.srt`, so I built one and decided to share it.

This app integrates with LM Studio, Ollama and OpenRouter to power its LLM translation engine. In the second tab, you’ll find a simple RAG (Retrieval-Augmented Generation) system that lets you extract specific information from a document—like character names—and inject it into the system prompt, ensuring consistency (for example, it won’t mistranslate names in different ways).

The included `setup.bat` script uses Python 3.11 (other versions haven’t been tested).

If you have ideas for new features or spot any issues, please open an issue or start a discussion in this repository!
