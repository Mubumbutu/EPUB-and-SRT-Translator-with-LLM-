from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLineEdit, 
    QPushButton, QTextEdit, QProgressBar, QFileDialog, QMessageBox, QGridLayout, 
    QDoubleSpinBox, QPlainTextEdit, QLabel, QSpinBox, QComboBox, QSplitter
)
from PyQt6.QtGui import QTextOption
import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import re
import tiktoken
import requests
import logging

# App Settings for storing last file path
@dataclass
class AppSettings:
    last_file_path: str = ""

class SettingsManager:
    SETTINGS_FILE = "remember_file.json"
    
    @staticmethod
    def load_settings() -> AppSettings:
        try:
            if os.path.exists(SettingsManager.SETTINGS_FILE):
                with open(SettingsManager.SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    return AppSettings(last_file_path=data.get('last_file_path', ""))
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
        return AppSettings()
    
    @staticmethod
    def save_settings(settings: AppSettings) -> None:
        try:
            with open(SettingsManager.SETTINGS_FILE, 'w') as f:
                json.dump({'last_file_path': settings.last_file_path}, f)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")

# File Processor for both EPUB and SRT
class FileProcessor:
    @staticmethod
    def load_document(file_path: str) -> Tuple[int, Optional[str]]:
        if file_path.lower().endswith('.epub'):
            try:
                book = epub.read_epub(file_path)
                chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
                return len(chapters), None
            except Exception as e:
                return 0, f"Error loading EPUB: {e}"
        elif file_path.lower().endswith('.srt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                blocks = content.split('\n\n')
                num_entries = len([block for block in blocks if block.strip()])
                return num_entries, None
            except Exception as e:
                return 0, f"Error loading SRT: {e}"
        else:
            return 0, "Unsupported file type"
    
    @staticmethod
    def convert_to_markdown(file_path: str, indices: List[int] = None) -> Tuple[str, Optional[str]]:
        if file_path.lower().endswith('.epub'):
            try:
                book = epub.read_epub(file_path)
                all_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
                if indices is not None:
                    selected = [all_items[i] for i in indices if 0 <= i < len(all_items)]
                else:
                    selected = all_items
                
                markdown_sections: List[str] = []
                for chap in selected:
                    raw = chap.get_content().decode('utf-8')
                    soup = BeautifulSoup(raw, 'html.parser')
                    paras = soup.find_all('p')
                    for p in paras:
                        text = p.get_text(strip=True)
                        if text:
                            markdown_sections.append(text)
                    markdown_sections.append('---')
                
                if markdown_sections and markdown_sections[-1] == '---':
                    markdown_sections.pop()
                
                return '\n\n'.join(markdown_sections), None
            except Exception as e:
                return "", f"Error converting EPUB: {e}"
        elif file_path.lower().endswith('.srt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                blocks = content.split('\n\n')
                subtitles = []
                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) >= 3:
                        text = ' '.join(lines[2:]).strip()
                        if text:
                            subtitles.append(text)
                return '\n\n'.join(subtitles), None
            except Exception as e:
                return "", f"Error converting SRT: {e}"
        else:
            return "", "Unsupported file type"

# File Processing Worker
class FileProcessingWorker(QThread):
    finished = pyqtSignal(str, list, str)

    def __init__(self, file_path: str, indices: List[int]):
        super().__init__()
        self.file_path = file_path
        self.indices = indices

    def run(self):
        try:
            full_md, err = FileProcessor.convert_to_markdown(self.file_path, self.indices)
            if err:
                self.finished.emit("", [], err)
                return
            paras = [p.strip() for p in full_md.split("\n\n") if p.strip() and p.strip() != "---"]
            self.finished.emit(full_md, paras, "")
        except Exception as e:
            self.finished.emit("", [], str(e))

# Token Counter
class TokenCounter:
    @staticmethod
    def count_tokens(text: str, encoding_name: str = 'cl100k_base') -> int:
        encoder = tiktoken.get_encoding(encoding_name)
        return len(encoder.encode(text))

# Enhanced Text Processor
class EnhancedTextProcessor:
    @staticmethod
    def semantic_only(
        full_text: str,
        question: str,
        min_similarity: float = 0.2,
        top_k: int = 5,
        window: int = 1
    ) -> List[Dict]:
        query_clean = question.lower().strip()
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
        results = []

        for idx, para in enumerate(paragraphs):
            sim = SequenceMatcher(None, query_clean, para.lower()).ratio()
            if sim >= min_similarity:
                results.append({
                    "paragraph_id": idx,
                    "text": para,
                    "match_type": "semantic",
                    "score": round(sim, 2)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        for hit in results:
            idx = hit["paragraph_id"]
            start = max(0, idx - window)
            end = min(len(paragraphs) - 1, idx + window)
            hit["context"] = "\n\n".join(paragraphs[start:end+1])

        return results

# Simple QA System
class SimpleQaSystem:
    @staticmethod
    def generate_answer(
        question: str,
        context: List[Dict],
        extra_instructions: str = "",
        api_url: str = "http://localhost:1234/v1/chat/completions",
        model_name: str = "local-model",
        temperature: float = 0.0
    ) -> Tuple[str, Dict]:
        if not context:
            return "No relevant information found in the document.", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        context_str = ""
        for entry in context:
            context_str += f"\n\n[Paragraph {entry['paragraph_id']}] {entry['text']}"

        prompt_parts = [
            f"**Question:**\n{question}",
            f"**Context:**{context_str}"
        ]
        if extra_instructions:
            prompt_parts.append(f"**User Instructions (priority):**\n{extra_instructions}")
        else:
            prompt_parts.append(
                "**Rules:**\n"
                "1. Be precise.\n"
                "2. If unsure, say \"I don't know.\"\n"
                "3. Mention paragraph numbers."
            )

        prompt = "\n\n".join(prompt_parts)
        prompt_tokens = TokenCounter.count_tokens(prompt)

        try:
            if "localhost:11434" in api_url or "ollama" in api_url.lower():
                answer = SimpleQaSystem._call_ollama_api(prompt, api_url, model_name, temperature)
            else:
                answer = SimpleQaSystem._call_lm_studio_api(prompt, api_url, temperature)

            completion_tokens = TokenCounter.count_tokens(answer)
            token_stats = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            return answer, token_stats

        except Exception as e:
            logging.error(f"LLM API error: {e}")
            return f"Error calling LLM API: {e}", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens
            }

    @staticmethod
    def _call_ollama_api(prompt: str, api_url: str, model_name: str, temperature: float) -> str:
        headers = {"Content-Type": "application/json"}
        base_url = api_url.replace('/api/generate', '').rstrip('/')
        try:
            test_response = requests.get(f"{base_url}", timeout=5)
            logging.debug(f"Ollama base URL response: {test_response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama server not responding at {base_url}. Run: ollama serve. Error: {e}")
        
        try:
            health_response = requests.get(f"{base_url}/api/tags", timeout=5)
            logging.debug(f"Ollama /api/tags response: {health_response.status_code}")
            if health_response.status_code != 200:
                raise Exception(f"Ollama /api/tags not responding. Status: {health_response.status_code}. Run: ollama serve")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama server not running. Run: ollama serve")
        except requests.exceptions.Timeout:
            raise Exception("Connection timeout to Ollama. Run: ollama serve")
        
        try:
            models_data = health_response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            logging.debug(f"Available models: {available_models}")
            if model_name not in available_models:
                model_base = model_name.split(':')[0]
                matching_models = [m for m in available_models if m.startswith(model_base)]
                if matching_models:
                    logging.warning(f"Using model: {matching_models[0]} instead of {model_name}")
                    model_name = matching_models[0]
                elif available_models:
                    logging.warning(f"Model '{model_name}' does not exist. Using available: {available_models[0]}")
                    model_name = available_models[0]
                else:
                    raise Exception(f"No models available in Ollama. Install a model: ollama pull llama3.2")
        except Exception as e:
            logging.warning(f"Cannot check models: {e}. Trying with provided model...")
        
        ollama_payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        generate_url = f"{base_url}/api/generate"
        logging.debug(f"Calling Ollama: {generate_url} with model: {model_name}")
        response = requests.post(generate_url, headers=headers, json=ollama_payload, timeout=None)
        logging.debug(f"Ollama response status: {response.status_code}")
        logging.debug(f"Ollama response headers: {dict(response.headers)}")
        if response.status_code != 200:
            logging.error(f"Ollama error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        if 'response' not in data:
            raise Exception(f"Invalid response from Ollama: {data}")
        return data.get('response', '').strip()

    @staticmethod
    def _call_lm_studio_api(prompt: str, api_url: str, temperature: float) -> str:
        headers = {"Content-Type": "application/json"}
        lm_studio_payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        response = requests.post(api_url, headers=headers, json=lm_studio_payload, timeout=None)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

# QA Worker
class QaWorker(QThread):
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)
    
    def __init__(self, markdown_text, question, min_sim, top_k, context_mode, custom_instr, snippet_length, 
                 api_url="http://localhost:1234/v1/chat/completions", model_name="local-model", temperature=0.0):
        super().__init__()
        self.markdown_text = markdown_text
        self.question = question
        self.min_sim = min_sim
        self.top_k = top_k
        self.context_mode = context_mode
        self.custom_instr = custom_instr
        self.snippet_length = snippet_length
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        
    def run(self):
        try:
            cm = self.context_mode
            if cm.startswith("Snippet"):
                snippet_mode = True
                window = 0
            elif "Surrounding" in cm:
                snippet_mode = False
                try:
                    window = int(cm.split()[-2])
                except (ValueError, IndexError):
                    window = 1
            elif cm.startswith("Full Paragraph"):
                snippet_mode = False
                window = 0
            else:
                snippet_mode = False
                window = 0
                
            relevant_sections = EnhancedTextProcessor.semantic_only(
                full_text=self.markdown_text,
                question=self.question,
                min_similarity=self.min_sim,
                top_k=self.top_k,
                window=window
            )
            
            if snippet_mode:
                llm_context = [
                    {
                        "paragraph_id": sec["paragraph_id"],
                        "text": sec["text"][:self.snippet_length],
                        "score": sec.get("score", 1.0)
                    }
                    for sec in relevant_sections
                ]
            else:
                llm_context = []
                for sec in relevant_sections:
                    block = sec.get("context", sec["text"])
                    llm_context.append({
                        "paragraph_id": sec["paragraph_id"],
                        "text": block,
                        "score": sec.get("score", 1.0)
                    })
                    
            answer, token_stats = SimpleQaSystem.generate_answer(
                question=self.question,
                context=llm_context,
                extra_instructions=self.custom_instr,
                api_url=self.api_url,
                model_name=self.model_name,
                temperature=self.temperature
            )
            
            result_text = f"Question: {self.question}\n\n"
            result_text += f"Answer:\n{answer}\n\n"
            result_text += "Related Paragraphs:\n"
            for sec in relevant_sections:
                result_text += (
                    f"\nParagraph {sec['paragraph_id']} "
                    f"(score: {sec.get('score', 1.0):.2f}):\n"
                    f"{sec['text']}\n"
                )
                
            result_text += f"\n\n--- Token Usage ---\n"
            result_text += f"Prompt Tokens: {token_stats['prompt_tokens']:,}\n"
            result_text += f"Completion Tokens: {token_stats['completion_tokens']:,}\n"
            result_text += f"Total Tokens: {token_stats['total_tokens']:,}\n"
            result_text += "~ Note: Token count is approximate and may vary by model."
            
            self.finished.emit(result_text, relevant_sections)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")

# Main Widget
class SmartQAWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.MIN_SIMILARITY_DEFAULT = 0.3
        self.TOP_K_DEFAULT = 50
        self.SNIPPET_LENGTH = 500
        self.parent_window = parent
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Vertical, self)

        upper_container = QWidget()
        upper_layout = QVBoxLayout(upper_container)
        upper_layout.setSpacing(10)
        upper_layout.setContentsMargins(0, 0, 0, 0)

        # File Selection Group
        file_group = QGroupBox("File Selection", upper_container)
        file_layout = QHBoxLayout()
        file_layout.setSpacing(8)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a file...")
        self.file_path_edit.textChanged.connect(self.on_file_path_changed)
        self.file_path_edit.returnPressed.connect(
            lambda: self.process_file() if self.process_btn.isEnabled() else None
        )
        file_layout.addWidget(self.file_path_edit, 1)

        clear_path_btn = QPushButton("×")
        clear_path_btn.setFixedWidth(24)
        clear_path_btn.setToolTip("Clear path")
        clear_path_btn.clicked.connect(lambda: self.file_path_edit.clear())
        file_layout.addWidget(clear_path_btn)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)

        file_group.setLayout(file_layout)
        upper_layout.addWidget(file_group)

        # Process Button and Status
        process_layout = QHBoxLayout()
        process_layout.setSpacing(8)

        self.process_btn = QPushButton("Process File")
        self.process_btn.clicked.connect(self.process_file)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn)

        self.status_label = QLabel("No file loaded")
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        process_layout.addWidget(self.status_label, 1)

        upper_layout.addLayout(process_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p% - %v/%m items processed")
        upper_layout.addWidget(self.progress_bar)

        # Search Settings Group
        settings_group = QGroupBox("Search Settings", upper_container)
        settings_layout = QGridLayout()
        settings_layout.setSpacing(8)
        settings_layout.setColumnStretch(1, 1)

        min_similarity_label = QLabel("Minimum Similarity:")
        min_similarity_label.setToolTip(
            "Minimum semantic similarity threshold (0.0–1.0). Higher → stricter matching."
        )
        settings_layout.addWidget(min_similarity_label, 0, 0)
        self.min_similarity_spin = QDoubleSpinBox()
        self.min_similarity_spin.setRange(0.0, 1.0)
        self.min_similarity_spin.setSingleStep(0.05)
        self.min_similarity_spin.setValue(self.MIN_SIMILARITY_DEFAULT)
        self.min_similarity_spin.setDecimals(2)
        self.min_similarity_spin.setSuffix("%")
        self.min_similarity_spin.setSpecialValueText("0% (All)")
        settings_layout.addWidget(self.min_similarity_spin, 0, 1)

        top_k_label = QLabel("Top K Results:")
        top_k_label.setToolTip("Maximum number of paragraphs to retrieve.")
        settings_layout.addWidget(top_k_label, 1, 0)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 999)
        self.top_k_spin.setValue(self.TOP_K_DEFAULT)
        self.top_k_spin.setSuffix(" paragraphs")
        settings_layout.addWidget(self.top_k_spin, 1, 1)

        self.snippet_length_label = QLabel("Snippet Length:")
        self.snippet_length_label.setToolTip("Number of characters to include in snippet mode.")
        settings_layout.addWidget(self.snippet_length_label, 2, 0)
        self.snippet_length_spin = QSpinBox()
        self.snippet_length_spin.setRange(50, 5000)
        self.snippet_length_spin.setValue(self.SNIPPET_LENGTH)
        self.snippet_length_spin.setSuffix(" characters")
        self.snippet_length_spin.valueChanged.connect(self.update_context_mode_items)
        settings_layout.addWidget(self.snippet_length_spin, 2, 1)

        context_mode_label = QLabel("Context Mode:")
        context_mode_label.setToolTip(
            "Snippet: Send only the first N characters of each paragraph.\n"
            "Full Paragraph: Send the entire paragraph.\n"
            "Full Paragraph + Surrounding X: Send the paragraph plus X paragraphs before/after."
        )
        settings_layout.addWidget(context_mode_label, 3, 0)
        self.context_mode_combo = QComboBox()
        settings_layout.addWidget(self.context_mode_combo, 3, 1)
        self.context_mode_combo.currentIndexChanged.connect(self.on_context_mode_changed)

        custom_prompt_label = QLabel("Additional Instructions:")
        custom_prompt_label.setToolTip(
            "These lines will be added to the constructed LLM prompt as custom instructions."
        )
        settings_layout.addWidget(custom_prompt_label, 4, 0, Qt.AlignmentFlag.AlignTop)
        self.custom_prompt_edit = QPlainTextEdit()
        self.custom_prompt_edit.setPlaceholderText(
            "Enter additional instructions for the LLM prompt...\n"
            "Example: 'Focus on technical details' or 'Explain in simple terms'"
        )
        self.custom_prompt_edit.setMaximumHeight(60)
        settings_layout.addWidget(self.custom_prompt_edit, 4, 1)

        settings_group.setLayout(settings_layout)
        upper_layout.addWidget(settings_group)

        # Question and Answer Input Group
        qa_input_group = QGroupBox("Ask Document", upper_container)
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)

        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText("Enter your question about the document...")
        self.question_edit.returnPressed.connect(
            lambda: self.handle_search() if self.search_btn.isEnabled() else None
        )
        input_layout.addWidget(self.question_edit)

        self.search_btn = QPushButton("Find Answer")
        self.search_btn.clicked.connect(self.handle_search)
        self.search_btn.setEnabled(False)
        input_layout.addWidget(self.search_btn)

        qa_input_group.setLayout(input_layout)
        upper_layout.addWidget(qa_input_group)

        splitter.addWidget(upper_container)

        # Results Group
        results_group = QGroupBox("Results", self)
        results_layout = QVBoxLayout()
        results_layout.setSpacing(8)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Results will appear here after processing your question...")
        self.results_text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.results_text.setAcceptRichText(True)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        splitter.addWidget(results_group)

        main_layout.addWidget(splitter)

        # Initialize internal state
        self.markdown_text = ''
        self.file_processed = False
        self.file_path = ''

        # Load last file path if available
        settings = SettingsManager.load_settings()
        if settings.last_file_path and os.path.exists(settings.last_file_path):
            self.file_path_edit.setText(settings.last_file_path)
            self.on_file_path_changed()

        self.update_context_mode_items()

    def update_context_mode_items(self):
        n_chars = self.snippet_length_spin.value()
        items = [
           f"Snippet (first {n_chars} characters)",
           "Full Paragraph",
           "Full Paragraph + Surrounding 1 paragraph",
           "Full Paragraph + Surrounding 2 paragraphs",
           "Full Paragraph + Surrounding 5 paragraphs",
           "Full Paragraph + Surrounding 10 paragraphs"
        ]
        self.context_mode_combo.clear()
        self.context_mode_combo.addItems(items)

    def on_context_mode_changed(self, index):
        is_fragment = (index == 0)
        self.snippet_length_label.setVisible(is_fragment)
        self.snippet_length_spin.setVisible(is_fragment)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "EPUB and SRT Files (*.epub *.srt);;All Files (*)"
        )
        if path:
            self.file_path_edit.setText(path)
            self.on_file_path_changed()

    def on_file_path_changed(self):
        path = self.file_path_edit.text().strip()
        is_valid_file = (path.lower().endswith('.epub') or path.lower().endswith('.srt')) and os.path.isfile(path)
        self.process_btn.setEnabled(is_valid_file)
        self.search_btn.setEnabled(False)

        if not path:
            self.status_label.setText("No file loaded")
            self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        elif is_valid_file:
            self.status_label.setText(f"Ready to process: {os.path.basename(path)}")
            self.status_label.setStyleSheet("color: #FFD700; font-style: normal;")
        else:
            self.status_label.setText("Invalid file selected")
            self.status_label.setStyleSheet("color: #8B0000; font-style: italic;")

    def process_file(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "Error", "Invalid file selected.")
            return

        try:
            section_count, error = FileProcessor.load_document(file_path)
            if error:
                QMessageBox.warning(self, "File Error", error)
                return
            settings = SettingsManager.load_settings()
            settings.last_file_path = file_path
            SettingsManager.save_settings(settings)
            self.file_path = file_path

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.process_btn.setEnabled(False)
            self.search_btn.setEnabled(False)

            if file_path.lower().endswith('.epub'):
                indices = list(range(section_count))
            else:  # SRT
                indices = None

            self.worker = FileProcessingWorker(file_path, indices)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def on_processing_finished(self, markdown: str, chunks: List[str], error: str):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

        if error:
            QMessageBox.critical(self, "Processing Error", error)
            self.status_label.setText("File processing failed")
            self.status_label.setStyleSheet("color: #8B0000; font-style: italic;")
            self.search_btn.setEnabled(False)
            return

        self.markdown_text = markdown
        self.search_btn.setEnabled(True)
        self.status_label.setText(f"File processed: {os.path.basename(self.file_path)}")
        self.status_label.setStyleSheet("color: #006400; font-style: normal;")

    def handle_search(self):
        question = self.question_edit.text().strip()
        if not question:
            QMessageBox.warning(self, "Error", "Please enter a question first.")
            return

        self.search_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet("color: #FFD700; font-style: italic;")
        QApplication.processEvents()

        min_sim = self.min_similarity_spin.value()
        top_k = self.top_k_spin.value()
        context_mode = self.context_mode_combo.currentText()
        custom_instr = self.custom_prompt_edit.toPlainText()

        win = self.window()
        llm_choice = win.app_settings.get("llm_choice", "LM Studio")
        model_name = win.app_settings.get("ollama_model_name", "").strip() if llm_choice == "Ollama" else \
                     win.app_settings.get("openrouter_model_name", "").strip() if llm_choice == "Openrouter" else \
                     "local-model"
        endpoint = win.app_settings.get("ollama_endpoint", "http://localhost:11434").strip()

        if llm_choice == "Ollama":
            api_url = endpoint.rstrip("/")
        elif llm_choice == "Openrouter":
            api_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            api_url = "http://localhost:1234/v1/chat/completions"

        if llm_choice == "Ollama" and not model_name:
            QMessageBox.warning(self, "Missing Model", "For Ollama, set the model name in Options.")
            self.search_btn.setEnabled(True)
            self.status_label.setText("")
            return
        elif llm_choice == "Openrouter" and (not win.app_settings.get("openrouter_api_key") or not model_name):
            QMessageBox.warning(self, "Missing Settings", "For Openrouter, provide API key and model name in Options.")
            self.search_btn.setEnabled(True)
            self.status_label.setText("")
            return

        temperature = 0.0

        self.qa_worker = QaWorker(
            self.markdown_text,
            question,
            min_sim,
            top_k,
            context_mode,
            custom_instr,
            self.SNIPPET_LENGTH,
            api_url,
            model_name,
            temperature
        )
        self.qa_worker.finished.connect(self.on_search_success)
        self.qa_worker.error.connect(self.on_search_error)
        self.qa_worker.start()

    def on_search_success(self, result_text, relevant_sections):
        self.results_text.setPlainText(result_text)
        self.status_label.setText("Completed")
        self.status_label.setStyleSheet("color: #006400; font-style: normal;")
        self.search_btn.setEnabled(True)

    def on_search_error(self, error_msg):
        self.status_label.setText("Error")
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", error_msg)