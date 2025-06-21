import sys
import json
import uuid
import re
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QTextEdit, QFileDialog, QLineEdit,
    QSplitter, QLabel, QSpinBox, QListWidgetItem, QMessageBox,
    QDoubleSpinBox, QCheckBox, QTabWidget, QComboBox, QProgressBar,
    QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import logging

from translation_worker import TranslationWorker
from epub_creator import EPUBCreator
from system_rag import SmartQAWidget

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SRTCreator(QThread):
    finished = pyqtSignal(str, bool)

    def __init__(self, paragraphs, output_path):
        super().__init__()
        self.paragraphs = paragraphs
        self.output_path = output_path

    def run(self):
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for para in self.paragraphs:
                    text = para['translated_text'] if para['is_translated'] else para['original_text']
                    f.write(f"{para['id']}\n")
                    f.write(f"{para['timestamp']}\n")
                    f.write(f"{text}\n\n")
            self.finished.emit(self.output_path, False)
        except Exception as e:
            self.finished.emit(str(e), True)

class TranslatorApp(QMainWindow):
    SOURCE_LANGUAGES = [
        ("Auto", None),
        ("Bulgarian", "BG"),
        ("Czech", "CS"),
        ("Danish", "DA"),
        ("German", "DE"),
        ("Greek", "EL"),
        ("English", "EN"),
        ("Spanish", "ES"),
        ("Estonian", "ET"),
        ("Finnish", "FI"),
        ("French", "FR"),
        ("Hungarian", "HU"),
        ("Indonesian", "ID"),
        ("Italian", "IT"),
        ("Japanese", "JA"),
        ("Korean", "KO"),
        ("Lithuanian", "LT"),
        ("Latvian", "LV"),
        ("Norwegian (Bokmål)", "NB"),
        ("Dutch", "NL"),
        ("Polish", "PL"),
        ("Portuguese", "PT"),
        ("Romanian", "RO"),
        ("Russian", "RU"),
        ("Slovak", "SK"),
        ("Slovenian", "SL"),
        ("Swedish", "SV"),
        ("Turkish", "TR"),
        ("Ukrainian", "UK"),
        ("Chinese", "ZH"),
    ]

    TARGET_LANGUAGES = [
        ("Bulgarian", "BG"),
        ("Czech", "CS"),
        ("Danish", "DA"),
        ("German", "DE"),
        ("Greek", "EL"),
        ("English", "EN"),
        ("English (British)", "EN-GB"),
        ("English (American)", "EN-US"),
        ("Spanish", "ES"),
        ("Estonian", "ET"),
        ("Finnish", "FI"),
        ("French", "FR"),
        ("Hungarian", "HU"),
        ("Indonesian", "ID"),
        ("Italian", "IT"),
        ("Japanese", "JA"),
        ("Korean", "KO"),
        ("Lithuanian", "LT"),
        ("Latvian", "LV"),
        ("Norwegian (Bokmål)", "NB"),
        ("Dutch", "NL"),
        ("Polish", "PL"),
        ("Portuguese", "PT"),
        ("Portuguese (Portugal)", "PT-PT"),
        ("Portuguese (Brazil)", "PT-BR"),
        ("Romanian", "RO"),
        ("Russian", "RU"),
        ("Slovak", "SK"),
        ("Slovenian", "SL"),
        ("Swedish", "SV"),
        ("Turkish", "TR"),
        ("Ukrainian", "UK"),
        ("Chinese", "ZH"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPUB and SRT Translator with LLM by Mubumbutu")
        self.setGeometry(100, 100, 1600, 900)
        
        self.book = None
        self.paragraphs = []
        self.original_file_path = None
        self.file_type = None
        
        self.app_settings = {}
        self.load_app_settings()

        self.full_prompts_visible = False
        self.custom_ollama_prompt = None
        self.custom_system_prompt = None  
        self.custom_user_prompt = None
        self.sync_in_progress = False
        
        self.init_ui()

    def init_ui(self):
        translator_widget = QWidget()
        translator_layout = QVBoxLayout(translator_widget)

        top_panel = QHBoxLayout()
        btn_open = QPushButton("Open File")
        btn_open.clicked.connect(self.open_file)
        btn_save_session = QPushButton("Save Session")
        btn_save_session.clicked.connect(self.save_session)
        btn_load_session = QPushButton("Load Session")
        btn_load_session.clicked.connect(self.load_session)
        top_panel.addWidget(btn_open)
        top_panel.addWidget(btn_save_session)
        top_panel.addWidget(btn_load_session)
        translator_layout.addLayout(top_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        filter_buttons = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(lambda: self.toggle_all_selection(True))
        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(lambda: self.toggle_all_selection(False))
        filter_buttons.addWidget(btn_select_all)
        filter_buttons.addWidget(btn_deselect_all)
        left_layout.addLayout(filter_buttons)

        select_mismatch_layout = QHBoxLayout()
        btn_select_untranslated = QPushButton("Select Untranslated")
        btn_select_untranslated.clicked.connect(lambda: self.toggle_selection_by_translated(False))
        btn_select_mismatch = QPushButton("Select Mismatch")
        btn_select_mismatch.clicked.connect(lambda: self.toggle_selection_mismatch(True))
        select_mismatch_layout.addWidget(btn_select_untranslated)
        select_mismatch_layout.addWidget(btn_select_mismatch)
        left_layout.addLayout(select_mismatch_layout)

        show_buttons = QHBoxLayout()
        btn_show_all = QPushButton("Show All")
        btn_show_all.clicked.connect(lambda: self.filter_list(None))
        btn_show_translated = QPushButton("Show Translated")
        btn_show_translated.clicked.connect(lambda: self.filter_list(True))
        btn_show_untranslated = QPushButton("Show Untranslated")
        btn_show_untranslated.clicked.connect(lambda: self.filter_list(False))
        btn_show_mismatch = QPushButton("Show Mismatch")
        btn_show_mismatch.clicked.connect(lambda: self.filter_mismatch(True))
        show_buttons.addWidget(btn_show_all)
        show_buttons.addWidget(btn_show_translated)
        show_buttons.addWidget(btn_show_untranslated)
        show_buttons.addWidget(btn_show_mismatch)
        left_layout.addLayout(show_buttons)

        search_layout = QHBoxLayout()
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItems(["Original", "Translation"])
        self.search_mode_combo.setToolTip("Search in original or translation")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search for word / phrase...")
        self.search_edit.textChanged.connect(self.filter_search)
        search_layout.addWidget(self.search_mode_combo)
        search_layout.addWidget(self.search_edit)
        left_layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.display_selected_fragment)
        left_layout.addWidget(self.list_widget)

        bottom_left_layout = QVBoxLayout()
        row_layout = QHBoxLayout()
        self.auto_fix_checkbox = QCheckBox("Auto-fix mismatch")
        self.auto_fix_checkbox.setToolTip("Automatically retry translation for mismatched fragments")
        row_layout.addWidget(self.auto_fix_checkbox)

        lbl_auto_fix_tries = QLabel("Number of attempts:")
        lbl_auto_fix_tries.setToolTip("How many times to retry translation in case of mismatch")
        row_layout.addWidget(lbl_auto_fix_tries)
        self.auto_fix_spinbox = QSpinBox()
        self.auto_fix_spinbox.setRange(1, 10)
        self.auto_fix_spinbox.setValue(3)
        row_layout.addWidget(self.auto_fix_spinbox)

        btn_cancel = QPushButton("Cancel Translation")
        btn_cancel.clicked.connect(self.cancel_translation)
        row_layout.addWidget(btn_cancel)
        bottom_left_layout.addLayout(row_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        bottom_left_layout.addWidget(self.progress_bar)

        left_layout.addLayout(bottom_left_layout)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        orig_container = QWidget()
        orig_layout = QVBoxLayout(orig_container)
        orig_layout.setContentsMargins(0, 0, 0, 0)
        orig_layout.setSpacing(4)
        lbl_original = QLabel("Original:")
        orig_layout.addWidget(lbl_original)
        self.original_text_view = QTextEdit()
        self.original_text_view.setReadOnly(True)
        self.original_text_view.setPlaceholderText("Original text will appear here.")
        orig_layout.addWidget(self.original_text_view)
        preview_splitter.addWidget(orig_container)

        trans_container = QWidget()
        trans_layout = QVBoxLayout(trans_container)
        trans_layout.setContentsMargins(0, 0, 0, 0)
        trans_layout.setSpacing(4)
        lbl_translated = QLabel("Translation (editable):")
        trans_layout.addWidget(lbl_translated)
        self.translated_text_view = QTextEdit()
        self.translated_text_view.setPlaceholderText("Translated sentence will appear here.")
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        trans_layout.addWidget(self.translated_text_view)

        deepl_layout = QHBoxLayout()
        btn_translate_deePL = QPushButton("Translate DeepL")
        btn_translate_deePL.clicked.connect(self.translate_with_deepl)
        deepl_layout.addWidget(btn_translate_deePL)

        self.deepl_mode_combo = QComboBox()
        self.deepl_mode_combo.addItems(["Free", "Pro"])
        deepl_layout.addWidget(self.deepl_mode_combo)

        self.source_lang_combo = QComboBox()
        for lang_name, lang_code in self.SOURCE_LANGUAGES:
            self.source_lang_combo.addItem(lang_name, lang_code)
        self.source_lang_combo.setCurrentIndex(self.source_lang_combo.findData("EN"))
        deepl_layout.addWidget(self.source_lang_combo)

        self.target_lang_combo = QComboBox()
        for lang_name, lang_code in self.TARGET_LANGUAGES:
            self.target_lang_combo.addItem(lang_name, lang_code)
        self.target_lang_combo.setCurrentIndex(self.target_lang_combo.findData("PL"))
        deepl_layout.addWidget(self.target_lang_combo)

        deepl_layout.addStretch()

        btn_check_mismatch = QPushButton("Check Mismatch")
        btn_check_mismatch.setStyleSheet("font-size: 14px; padding: 8px 12px;")
        btn_check_mismatch.clicked.connect(self.check_mismatch)
        btn_check_mismatch.setToolTip("Check mismatches between original and translation")
        deepl_layout.addWidget(btn_check_mismatch)

        trans_layout.addLayout(deepl_layout)
        preview_splitter.addWidget(trans_container)

        right_layout.addWidget(preview_splitter)

        llm_options_layout = QVBoxLayout()
        label_sys = QLabel("SYSTEM Instruction for LLM:")
        llm_options_layout.addWidget(label_sys)
        self.llm_system_prompt = QTextEdit()
        self.llm_system_prompt.setFixedHeight(250)
        self.llm_system_prompt.setPlainText(
            "You are a professional translator from English to Polish. Translate the text according to the following rules:\n"
            "1. Return ONLY the translated text – do not add comments, explanations, or extra blank lines.\n"
            "2. Preserve exactly all elements from the original:\n"
            "   - Quotation marks (\"...\", „...\") and apostrophes ('...') along with their positions.\n"
            "   - Punctuation marks without changing their number/position.\n"
            "   - Paragraph breaks and text structure.\n"
            "3. Prioritize fidelity to the meaning and intent of the author while maintaining natural Polish language. Convey tone, rhythm, and mood through appropriate word choice and sentence construction. Adapt metaphors, imagery, and wordplay, replacing idioms and cultural references with Polish equivalents of similar expressive power. Preserve humorous effects where present. Ensure terminological consistency and adapt style to the text genre (prose, poetry, fantasy). Create neologisms according to Polish word-formation logic. Ensure fluidity of the narrative from the perspective of a Polish reader, accepting natural text lengthening due to linguistic differences.\n"
            "4. Translate proper names as per the examples provided:\n"
            "Tadeusz → Tadek\n"
            "Other names: if no example is given, retain the original."
        )
        self.llm_system_prompt.textChanged.connect(self.on_main_system_prompt_changed)
        llm_options_layout.addWidget(self.llm_system_prompt)

        context_layout = QHBoxLayout()
        lbl_context = QLabel("Number of context paragraphs:")
        lbl_context.setToolTip("How many previous paragraphs to include as context")
        context_layout.addWidget(lbl_context)
        self.context_spinbox = QSpinBox()
        self.context_spinbox.setRange(0, 99999)
        self.context_spinbox.setValue(3)
        context_layout.addWidget(self.context_spinbox)
        llm_options_layout.addLayout(context_layout)

        temp_layout = QHBoxLayout()
        lbl_temp = QLabel("LLM Temperature:")
        lbl_temp.setToolTip("0.0 - deterministic, 1.0 - random responses")
        temp_layout.addWidget(lbl_temp)
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 1.0)
        self.temperature_spinbox.setSingleStep(0.05)
        self.temperature_spinbox.setValue(0.8)
        temp_layout.addWidget(self.temperature_spinbox)
        llm_options_layout.addLayout(temp_layout)
        right_layout.addLayout(llm_options_layout)

        action_buttons_layout = QHBoxLayout()
        btn_translate = QPushButton("Translate Selected")
        btn_translate.setStyleSheet("font-weight: bold; padding: 10px;")
        btn_translate.clicked.connect(self.start_translation)
        btn_show_full_prompts = QPushButton("Show Full LLM Instructions")
        btn_show_full_prompts.setStyleSheet("font-weight: bold; padding: 10px;")
        btn_show_full_prompts.clicked.connect(self.toggle_full_prompts_view)
        btn_save_file = QPushButton("Save as New File")
        btn_save_file.setStyleSheet("font-weight: bold; padding: 10px;")
        btn_save_file.clicked.connect(self.save_file)
        action_buttons_layout.addWidget(btn_translate)
        action_buttons_layout.addWidget(btn_show_full_prompts)
        action_buttons_layout.addWidget(btn_save_file)
        right_layout.addLayout(action_buttons_layout)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        translator_layout.addWidget(splitter)

        qa_widget = SmartQAWidget()
        options_widget = self.init_options_tab()

        tab_widget = QTabWidget()
        tab_widget.addTab(translator_widget, "Translator")
        tab_widget.addTab(qa_widget, "RAG System")
        tab_widget.addTab(options_widget, "Options")

        self.setCentralWidget(tab_widget)

    def init_options_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form_layout = QFormLayout()

        llm_label = QLabel("Select LLM:")
        self.llm_choice_combo = QComboBox()
        self.llm_choice_combo.addItems(["LM Studio", "Ollama", "Openrouter"])
        self.llm_choice_combo.currentTextChanged.connect(self.update_model_name_visibility)
        form_layout.addRow(llm_label, self.llm_choice_combo)

        self.ollama_model_label = QLabel("Ollama Model Name:")
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setPlaceholderText("e.g., llama3.2:3b")
        form_layout.addRow(self.ollama_model_label, self.ollama_model_edit)

        self.openrouter_api_key_label = QLabel("Openrouter API Key:")
        self.openrouter_api_key_edit = QLineEdit()
        self.openrouter_api_key_edit.setPlaceholderText("Enter your Openrouter API key")
        form_layout.addRow(self.openrouter_api_key_label, self.openrouter_api_key_edit)

        self.openrouter_model_label = QLabel("Openrouter Model Name:")
        self.openrouter_model_edit = QLineEdit()
        self.openrouter_model_edit.setPlaceholderText("e.g., openai/gpt-4")
        form_layout.addRow(self.openrouter_model_label, self.openrouter_model_edit)

        self.deepl_free_api_key_label = QLabel("DeepL Free API Key:")
        self.deepl_free_api_key_edit = QLineEdit()
        self.deepl_free_api_key_edit.setPlaceholderText("Enter your DeepL Free API key")
        form_layout.addRow(self.deepl_free_api_key_label, self.deepl_free_api_key_edit)

        self.deepl_pro_api_key_label = QLabel("DeepL Pro API Key:")
        self.deepl_pro_api_key_edit = QLineEdit()
        self.deepl_pro_api_key_edit.setPlaceholderText("Enter your DeepL Pro API key")
        form_layout.addRow(self.deepl_pro_api_key_label, self.deepl_pro_api_key_edit)

        layout.addLayout(form_layout)

        btn_save_options = QPushButton("Save Settings")
        btn_save_options.clicked.connect(self.save_app_settings)
        layout.addWidget(btn_save_options, alignment=Qt.AlignmentFlag.AlignRight)

        current_llm = self.app_settings.get("llm_choice", "LM Studio")
        self.llm_choice_combo.setCurrentText(current_llm)
        self.ollama_model_edit.setText(self.app_settings.get("ollama_model_name", ""))
        self.openrouter_api_key_edit.setText(self.app_settings.get("openrouter_api_key", ""))
        self.openrouter_model_edit.setText(self.app_settings.get("openrouter_model_name", ""))
        self.deepl_free_api_key_edit.setText(self.app_settings.get("deepl_free_api_key", ""))
        self.deepl_pro_api_key_edit.setText(self.app_settings.get("deepl_pro_api_key", ""))

        self.update_model_name_visibility(current_llm)
        return widget

    def update_model_name_visibility(self, llm_choice):
        is_ollama = llm_choice == "Ollama"
        is_openrouter = llm_choice == "Openrouter"
        self.ollama_model_label.setVisible(is_ollama)
        self.ollama_model_edit.setVisible(is_ollama)
        self.openrouter_api_key_label.setVisible(is_openrouter)
        self.openrouter_api_key_edit.setVisible(is_openrouter)
        self.openrouter_model_label.setVisible(is_openrouter)
        self.openrouter_model_edit.setVisible(is_openrouter)

    def save_app_settings(self):
        try:
            with open("app_settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
        except FileNotFoundError:
            settings = {}
        except Exception as e:
            QMessageBox.warning(self, "Load Warning", f"Could not load existing settings: {e}")
            settings = {}

        defaults = {
            "llm_choice": "LM Studio",
            "ollama_model_name": "",
            "openrouter_api_key": "",
            "openrouter_model_name": "",
            "ollama_endpoint": "http://localhost:11434",
            "deepl_free_api_key": "",
            "deepl_pro_api_key": ""
        }
        for key, default_val in defaults.items():
            settings.setdefault(key, default_val)

        settings["llm_choice"] = self.llm_choice_combo.currentText()
        if settings["llm_choice"] == "Ollama":
            settings["ollama_model_name"] = self.ollama_model_edit.text()
        elif settings["llm_choice"] == "Openrouter":
            settings["openrouter_api_key"] = self.openrouter_api_key_edit.text()
            settings["openrouter_model_name"] = self.openrouter_model_edit.text()

        settings["deepl_free_api_key"] = self.deepl_free_api_key_edit.text()
        settings["deepl_pro_api_key"] = self.deepl_pro_api_key_edit.text()

        try:
            with open("app_settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
            self.app_settings = settings
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")

    def load_app_settings(self):
        try:
            with open("app_settings.json", "r", encoding="utf-8") as f:
                self.app_settings = json.load(f)
        except FileNotFoundError:
            self.app_settings = {}
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load settings: {e}")
            self.app_settings = {}
        
        defaults = {
            "llm_choice": "LM Studio",
            "ollama_model_name": "",
            "openrouter_api_key": "",
            "openrouter_model_name": "",
            "ollama_endpoint": "http://localhost:11434",
            "deepl_free_api_key": "",
            "deepl_pro_api_key": ""
        }
        for key, val in defaults.items():
            self.app_settings.setdefault(key, val)
        
        try:
            with open("app_settings.json", "w", encoding="utf-8") as f:
                json.dump(self.app_settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def translate_with_deepl(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            self.show_message("No Selection", "Please select a fragment to translate.", QMessageBox.Icon.Warning)
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)
        original_text = self.paragraphs[idx]['original_text']

        mode = self.deepl_mode_combo.currentText()
        if mode == "Free":
            api_key = self.app_settings.get("deepl_free_api_key", "")
            endpoint = "https://api-free.deepl.com/v2/translate"
        elif mode == "Pro":
            api_key = self.app_settings.get("deepl_pro_api_key", "")
            endpoint = "https://api.deepl.com/v2/translate"
        else:
            self.show_message("Invalid Mode", "Selected mode is invalid.", QMessageBox.Icon.Critical)
            return

        if not api_key:
            self.show_message("Missing API Key", f"Please set the DeepL {mode} API key in Options.", QMessageBox.Icon.Warning)
            return

        source_lang = self.source_lang_combo.currentData()
        target_lang = self.target_lang_combo.currentData()
        if not target_lang:
            self.show_message("Missing Target Language", "Please select a target language.", QMessageBox.Icon.Warning)
            return

        headers = {
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "text": original_text,
            "target_lang": target_lang,
        }
        if source_lang:
            data["source_lang"] = source_lang

        try:
            response = requests.post(endpoint, headers=headers, data=data)
            response.raise_for_status()
            result = response.json()
            translated_text = result["translations"][0]["text"]

            self.paragraphs[idx]['translated_text'] = translated_text
            self.paragraphs[idx]['is_translated'] = True
            self.update_item_visuals(current_item, self.paragraphs[idx])

            self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
            self.translated_text_view.setText(translated_text)
            self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        except requests.exceptions.RequestException as e:
            try:
                response_text = response.text if 'response' in locals() else str(e)
                if response.status_code == 403:
                    error_msg = "Invalid API key. Please check your DeepL API key."
                elif response.status_code == 456:
                    error_msg = "Quota exceeded. Please check your DeepL account."
                else:
                    error_msg = f"DeepL API error: {response.status_code} - {response_text}"
            except NameError:
                error_msg = f"DeepL API error: {e}"
            self.show_message("Translation Error", error_msg, QMessageBox.Icon.Critical)
            self.paragraphs[idx]['translated_text'] = "Translation failed"
            self.paragraphs[idx]['is_translated'] = False
            self.update_item_visuals(current_item, self.paragraphs[idx])
            self.translated_text_view.setText("Translation failed")

    def toggle_full_prompts_view(self):
        if not hasattr(self, 'full_prompts_container'):
            self.create_full_prompts_container()
            self.full_prompts_visible = True
            self.full_prompts_container.setVisible(True)
            return

        self.full_prompts_visible = not self.full_prompts_visible
        self.full_prompts_container.setVisible(self.full_prompts_visible)
        if self.full_prompts_visible:
            self.update_full_prompts_content()

    def create_full_prompts_container(self):
        parent_widget = self.llm_system_prompt.parent()
        parent_layout = parent_widget.layout()
        
        self.full_prompts_container = QWidget()
        container_layout = QVBoxLayout(self.full_prompts_container)
        
        label = QLabel("Full instructions sent to LLM (editable):")
        label.setStyleSheet("font-weight: bold; color: #0066cc;")
        container_layout.addWidget(label)
        
        self.prompts_content_widget = QWidget()
        self.prompts_content_layout = QVBoxLayout(self.prompts_content_widget)
        container_layout.addWidget(self.prompts_content_widget)
        
        temp_spinbox_index = -1
        for i in range(parent_layout.count()):
            item = parent_layout.itemAt(i)
            if item and item.layout():
                layout = item.layout()
                for j in range(layout.count()):
                    widget = layout.itemAt(j).widget() if layout.itemAt(j) else None
                    if widget == self.temperature_spinbox:
                        temp_spinbox_index = i
                        break
                if temp_spinbox_index >= 0:
                    break
        
        if temp_spinbox_index >= 0:
            parent_layout.insertWidget(temp_spinbox_index, self.full_prompts_container)
        else:
            parent_layout.addWidget(self.full_prompts_container)
        
        self.full_prompts_container.setVisible(False)
        self.update_full_prompts_content()

    def update_full_prompts_content(self):
        if not hasattr(self, 'prompts_content_widget'):
            return
            
        for i in reversed(range(self.prompts_content_layout.count())):
            self.prompts_content_layout.itemAt(i).widget().setParent(None)
        
        llm_choice = self.app_settings.get("llm_choice", "LM Studio")
        
        if llm_choice == "Ollama":
            self.ollama_prompt_edit = QTextEdit()
            self.ollama_prompt_edit.setFixedHeight(200)
            prompt_text = self.custom_ollama_prompt if self.custom_ollama_prompt else (
                self.llm_system_prompt.toPlainText().strip() + "\n\n"
                "Context (ONLY for understanding, DO NOT translate):\n"
                "{context}\n---\n"
                "Translate ONLY this (do not write anything else):\n{core_text}"
            )
            self.ollama_prompt_edit.setPlainText(prompt_text)
            self.ollama_prompt_edit.textChanged.connect(self.on_ollama_prompt_changed)
            self.prompts_content_layout.addWidget(QLabel("Full prompt for Ollama:"))
            self.prompts_content_layout.addWidget(self.ollama_prompt_edit)
        else:
            splitter = QSplitter(Qt.Orientation.Horizontal)
            system_container = QWidget()
            system_layout = QVBoxLayout(system_container)
            system_layout.addWidget(QLabel("System prompt:"))
            self.system_prompt_edit = QTextEdit()
            self.system_prompt_edit.setFixedHeight(200)
            system_text = self.custom_system_prompt if self.custom_system_prompt else (
                self.llm_system_prompt.toPlainText().strip() + "\n\n"
                "Context (ONLY for understanding, DO NOT translate):\n"
                "{context}\n---"
            )
            self.system_prompt_edit.setPlainText(system_text)
            self.system_prompt_edit.textChanged.connect(self.on_system_prompt_changed)
            system_layout.addWidget(self.system_prompt_edit)
            splitter.addWidget(system_container)
            
            user_container = QWidget()
            user_layout = QVBoxLayout(user_container)
            user_layout.addWidget(QLabel("User prompt:"))
            self.user_prompt_edit = QTextEdit()
            self.user_prompt_edit.setFixedHeight(200)
            user_text = self.custom_user_prompt if self.custom_user_prompt else "Translate ONLY this:\n{core_text}"
            self.user_prompt_edit.setPlainText(user_text)
            self.user_prompt_edit.textChanged.connect(self.on_user_prompt_changed)
            user_layout.addWidget(self.user_prompt_edit)
            splitter.addWidget(user_container)
            self.prompts_content_layout.addWidget(splitter)

    def on_ollama_prompt_changed(self):
        if self.sync_in_progress or not hasattr(self, 'ollama_prompt_edit'):
            return
        self.custom_ollama_prompt = self.ollama_prompt_edit.toPlainText()
        self.extract_system_prompt_from_ollama()

    def on_system_prompt_changed(self):
        if self.sync_in_progress or not hasattr(self, 'system_prompt_edit'):
            return
        self.custom_system_prompt = self.system_prompt_edit.toPlainText()
        self.extract_system_prompt_from_lm_studio()

    def on_user_prompt_changed(self):
        if self.sync_in_progress or not hasattr(self, 'user_prompt_edit'):
            return
        self.custom_user_prompt = self.user_prompt_edit.toPlainText()

    def on_main_system_prompt_changed(self):
        if self.sync_in_progress:
            return
        self.sync_in_progress = True
        self.custom_ollama_prompt = None
        self.custom_system_prompt = None
        self.custom_user_prompt = None
        if hasattr(self, 'full_prompts_container') and self.full_prompts_visible:
            self.update_full_prompts_content()
        self.sync_in_progress = False

    def extract_system_prompt_from_ollama(self):
        if not self.custom_ollama_prompt:
            return
        prompt_text = self.custom_ollama_prompt
        context_marker = "Context (ONLY for understanding, DO NOT translate):"
        if context_marker in prompt_text:
            system_part = prompt_text.split(context_marker)[0].strip()
            if system_part and system_part != self.llm_system_prompt.toPlainText().strip():
                self.sync_in_progress = True
                self.llm_system_prompt.setPlainText(system_part)
                self.sync_in_progress = False

    def extract_system_prompt_from_lm_studio(self):
        if not self.custom_system_prompt:
            return
        prompt_text = self.custom_system_prompt
        context_marker = "Context (ONLY for understanding, DO NOT translate):"
        if context_marker in prompt_text:
            system_part = prompt_text.split(context_marker)[0].strip()
            if system_part and system_part != self.llm_system_prompt.toPlainText().strip():
                self.sync_in_progress = True
                self.llm_system_prompt.setPlainText(system_part)
                self.sync_in_progress = False

    def show_message(self, title, message, icon=QMessageBox.Icon.Information):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.exec()

    def filter_search(self):
        phrase = self.search_edit.text().lower().strip()
        mode = self.search_mode_combo.currentText()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            para = self.paragraphs[i]
            text = para['original_text'] if mode == "Original" else para['translated_text']
            visible = True if not phrase else (phrase in text.lower())
            item.setHidden(not visible)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Files (*.epub *.srt);;EPUB Files (*.epub);;SRT Files (*.srt)")
        if not path:
            return
        if path.lower().endswith('.epub'):
            self.file_type = "epub"
            self.load_epub(path)
        elif path.lower().endswith('.srt'):
            self.file_type = "srt"
            self.load_srt(path)
        else:
            self.show_message("Unsupported Format", "Selected file has an unsupported format.", QMessageBox.Icon.Warning)

    def load_epub(self, path):
        try:
            self.original_file_path = path
            self.book = epub.read_epub(path)
            self.paragraphs = []

            # Lista tagów do ekstrakcji - tylko blokowe elementy
            TAGS_TO_EXTRACT = [
                "h1","h2","h3","h4","h5","h6",
                "p","li","td","th","blockquote","pre"
            ]

            # Zbiór do unikania duplikatów: (item_href, czysty tekst)
            seen = set()

            for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                raw = item.get_content()
                if not raw:
                    continue
                html = raw.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')

                for tag_name in TAGS_TO_EXTRACT:
                    for elem in soup.find_all(tag_name):
                        clean_text = elem.get_text(separator=" ", strip=True)
                        if not clean_text:
                            continue
                        key = (item.get_name(), clean_text)
                        if key in seen:
                            continue
                        seen.add(key)

                        # Dodaj id, jeśli brak
                        if not elem.has_attr("id"):
                            elem["id"] = f"trans_{uuid.uuid4()}"
                            
                        # ZAPISZ ORYGINALNY HTML ELEMENTU
                        original_html = str(elem)

                        self.paragraphs.append({
                            "id": elem["id"],
                            "original_text": clean_text,
                            "translated_text": "",
                            "is_translated": False,
                            "item_href": item.get_name(),
                            "element_type": tag_name,
                            "original_html": original_html  # NOWE POLE
                        })

                # Zapisz zmienione id
                item.set_content(str(soup).encode('utf-8'))

            self.populate_list()
            self.show_message(
                "Success",
                f"Załadowano {len(self.paragraphs)} unikalnych fragmentów do tłumaczenia."
            )

        except Exception as e:
            self.show_message(
                "EPUB Load Error",
                f"Nie udało się wczytać pliku EPUB:\n{e}",
                QMessageBox.Icon.Critical
            )


    def load_srt(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
            self.paragraphs = []
            self.original_file_path = path
            for block in blocks:
                lines = block.split('\n')
                if len(lines) < 3:
                    continue
                number = lines[0].strip()
                timestamp = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                self.paragraphs.append({
                    'id': number,
                    'original_text': text,
                    'translated_text': '',
                    'is_translated': False,
                    'item_href': path,
                    'element_type': 'subtitle',
                    'timestamp': timestamp
                })
            self.populate_list()
            self.show_message("Success", f"Loaded {len(self.paragraphs)} subtitles for translation.")
        except Exception as e:
            self.show_message("SRT Load Error", f"Failed to load SRT file: {e}", QMessageBox.Icon.Critical)

    def populate_list(self):
        self.list_widget.clear()
        for i, para in enumerate(self.paragraphs):
            item = QListWidgetItem(f"Fragment {i+1}: {para['original_text'][:70]}...")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_widget.addItem(item)
            self.update_item_visuals(item, para)

    def toggle_selection_by_translated(self, translated: bool):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            if self.paragraphs[idx]['is_translated'] == translated:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _has_mismatch(self, idx: int) -> bool:
        para = self.paragraphs[idx]
        if not para.get('is_translated'):
            return False
        
        orig = para['original_text']
        trans = para['translated_text']
        
        def count_paragraphs(text: str) -> int:
            parts = [p for p in text.split('\n\n') if p.strip()]
            if len(parts) > 1:
                return len(parts)
            return len([p for p in text.split('\n') if p.strip()])
        
        paragraph_mismatch = (count_paragraphs(orig) != count_paragraphs(trans))
        
        def first_char_type(text):
            m = re.search(r'\S', text)
            if not m: 
                return "none"
            c = text[m.start()]
            return "digit" if c.isdigit() else "alpha" if c.isalpha() else "other"
        
        char_mismatch = (first_char_type(orig) != first_char_type(trans))
        
        # Nowa funkcja - sprawdzanie ostatniego znaku
        def last_char_type(text):
            m = re.search(r'\S(?=\s*$)', text)  # ostatni niepusty znak
            if not m:
                return "none"
            c = text[m.start()]
            if c in '.!?':
                return "sentence_end"
            elif c in ',;:':
                return "punctuation"
            elif c.isdigit():
                return "digit"
            elif c.isalpha():
                return "alpha"
            else:
                return "other"
        
        last_char_mismatch = (last_char_type(orig) != last_char_type(trans))
        
        def extract_placeholders(text):
            return set(re.findall(r'\{.*?\}|%s|%d', text))
        
        placeholder_mismatch = (extract_placeholders(orig) != extract_placeholders(trans))
        
        length_mismatch = orig and trans and (abs(len(orig) - len(trans)) > 0.5 * max(len(orig), len(trans)))
        
        def extract_numbers(text):
            return set(re.findall(r'\d+', text))
        
        num_mismatch = (extract_numbers(orig) != extract_numbers(trans))
        
        # Nowe sprawdzenia:
        
        # 1. Sprawdzanie formatowania (bold, italic, markdown)
        def extract_formatting(text):
            formatting = set()
            formatting.update(re.findall(r'\*\*.*?\*\*', text))  # bold
            formatting.update(re.findall(r'\*.*?\*', text))      # italic
            formatting.update(re.findall(r'`.*?`', text))        # code
            formatting.update(re.findall(r'_.*?_', text))        # underline
            return formatting
        
        formatting_mismatch = (extract_formatting(orig) != extract_formatting(trans))
        
        # 2. Sprawdzanie linków i URL-i
        def extract_urls(text):
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            markdown_links = re.findall(r'\[.*?\]\(.*?\)', text)
            urls = re.findall(url_pattern, text)
            return set(urls + markdown_links)
        
        url_mismatch = (extract_urls(orig) != extract_urls(trans))
        
        # 3. Sprawdzanie cudzysłowów i nawiasów
        def count_brackets_quotes(text):
            # 1) ignorujemy apostrofy wewnątrz słów (kontrakcje typu would'n't)
            filtered = re.sub(r"(?<=\w)'(?=\w)", "", text)
            # 2) zliczamy cytaty i nawiasy w przefiltrowanym tekście
            return {
                'quotes': filtered.count('"') + filtered.count("'"),
                'parentheses': filtered.count('(') + filtered.count(')'),
                'square_brackets': filtered.count('[') + filtered.count(']'),
                'curly_brackets': filtered.count('{') + filtered.count('}')
            }
        
        brackets_quotes_mismatch = (count_brackets_quotes(orig) != count_brackets_quotes(trans))
        
        # 4. Sprawdzanie wielkich liter na początku zdań
        def count_sentence_starts(text):
            sentences = re.split(r'[.!?]+\s+', text)
            caps_count = 0
            for sentence in sentences:
                if sentence.strip() and sentence.strip()[0].isupper():
                    caps_count += 1
            return caps_count
        
        sentence_caps_mismatch = abs(count_sentence_starts(orig) - count_sentence_starts(trans)) > 1
        
        # 5. Sprawdzanie emotikon i specjalnych symboli
        def extract_special_chars(text):
            # Emotikonki, symbole, znaki specjalne
            special_pattern = r'[😀-🙏🌀-🗿💀-🟿]|:\)|:\(|:D|;-?\)|:-?\(|:-?D'
            symbols = set(re.findall(special_pattern, text))
            # Dodaj inne symbole
            other_symbols = set(re.findall(r'[©®™§¶†‡•…‰′″‹›«»¡¿]', text))
            return symbols.union(other_symbols)
        
        special_chars_mismatch = (extract_special_chars(orig) != extract_special_chars(trans))
        
        # 6. Sprawdzanie list i numeracji
        def has_list_structure(text):
            patterns = [
                r'^\s*\d+\.',  # 1. 2. 3.
                r'^\s*[a-zA-Z]\.',  # a. b. c.
                r'^\s*[-*•]',  # bullet points
                r'^\s*\([a-zA-Z0-9]+\)'  # (1) (a) (i)
            ]
            lines = text.split('\n')
            for pattern in patterns:
                if sum(1 for line in lines if re.match(pattern, line)) >= 2:
                    return True
            return False
        
        list_structure_mismatch = (has_list_structure(orig) != has_list_structure(trans))
        
        return any([
            paragraph_mismatch, 
            char_mismatch, 
            last_char_mismatch,  # Twoja propozycja
            placeholder_mismatch, 
            length_mismatch, 
            num_mismatch,
            formatting_mismatch,
            url_mismatch,
            brackets_quotes_mismatch,
            sentence_caps_mismatch,
            special_chars_mismatch,
            list_structure_mismatch
        ])

    def toggle_selection_mismatch(self, select: bool):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            mismatch = self._has_mismatch(idx)
            if mismatch == select:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

    def update_item_visuals(self, item: QListWidgetItem, para_data: dict):
        idx = item.data(Qt.ItemDataRole.UserRole)
        orig = para_data.get('original_text', '')
        trans = para_data.get('translated_text', '') if para_data.get('is_translated') else ''
        is_translated = para_data.get('is_translated', False)
        mismatch = self._has_mismatch(idx)
        
        font = item.font()
        
        if is_translated:
            def count_paragraphs(text: str) -> int:
                parts = [p for p in text.split('\n\n') if p.strip()]
                if len(parts) > 1: 
                    return len(parts)
                return len([p for p in text.split('\n') if p.strip()])
            
            def first_char_type(text: str) -> str:
                m = re.search(r'\S', text)
                if not m: 
                    return "none"
                c = text[m.start()]
                if c.isdigit(): 
                    return "digit"
                if c.isalpha(): 
                    return "alpha"
                return "other"
            
            def last_char_type(text):
                m = re.search(r'\S(?=\s*$)', text)  # ostatni niepusty znak
                if not m:
                    return "none"
                c = text[m.start()]
                if c in '.!?':
                    return "sentence_end"
                elif c in ',;:':
                    return "punctuation"
                elif c.isdigit():
                    return "digit"
                elif c.isalpha():
                    return "alpha"
                else:
                    return "other"
            
            paragraph_mismatch = (count_paragraphs(orig) != count_paragraphs(trans))
            char_mismatch = (first_char_type(orig) != first_char_type(trans))
            last_char_mismatch = (last_char_type(orig) != last_char_type(trans))
            
            font.setUnderline(paragraph_mismatch)
            font.setItalic(char_mismatch)
            font.setStrikeOut(last_char_mismatch)  # Nowy styl dla ostatniego znaku
        else:
            font.setUnderline(False)
            font.setItalic(False)
            font.setStrikeOut(False)
        
        item.setFont(font)
        
        if mismatch:
            item.setForeground(QColor("red"))
        elif is_translated:
            item.setForeground(QColor("#228B22"))
        else:
            item.setForeground(QColor("white"))
        
        if mismatch:
            def extract_placeholders(text):
                return set(re.findall(r'\{.*?\}|%s|%d', text))
            
            def extract_numbers(text):
                return set(re.findall(r'\d+', text))
            
            def extract_formatting(text):
                formatting = set()
                formatting.update(re.findall(r'\*\*.*?\*\*', text))  # bold
                formatting.update(re.findall(r'\*.*?\*', text))      # italic
                formatting.update(re.findall(r'`.*?`', text))        # code
                formatting.update(re.findall(r'_.*?_', text))        # underline
                return formatting
            
            def extract_urls(text):
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                markdown_links = re.findall(r'\[.*?\]\(.*?\)', text)
                urls = re.findall(url_pattern, text)
                return set(urls + markdown_links)
            
            def count_brackets_quotes(text):
                return {
                    'quotes': text.count('"') + text.count("'") + text.count('"') + text.count('"'),
                    'parentheses': text.count('(') + text.count(')'),
                    'square_brackets': text.count('[') + text.count(']'),
                    'curly_brackets': text.count('{') + text.count('}')
                }
            
            def count_sentence_starts(text):
                sentences = re.split(r'[.!?]+\s+', text)
                caps_count = 0
                for sentence in sentences:
                    if sentence.strip() and sentence.strip()[0].isupper():
                        caps_count += 1
                return caps_count
            
            def extract_special_chars(text):
                # Emotikonki, symbole, znaki specjalne
                special_pattern = r'[😀-🙏🌀-🗿💀-🟿]|:\)|:\(|:D|;-?\)|:-?\(|:-?D'
                symbols = set(re.findall(special_pattern, text))
                # Dodaj inne symbole
                other_symbols = set(re.findall(r'[©®™§¶†‡•…‰′″‹›«»¡¿]', text))
                return symbols.union(other_symbols)
            
            def has_list_structure(text):
                patterns = [
                    r'^\s*\d+\.',  # 1. 2. 3.
                    r'^\s*[a-zA-Z]\.',  # a. b. c.
                    r'^\s*[-*•]',  # bullet points
                    r'^\s*\([a-zA-Z0-9]+\)'  # (1) (a) (i)
                ]
                lines = text.split('\n')
                for pattern in patterns:
                    if sum(1 for line in lines if re.match(pattern, line)) >= 2:
                        return True
                return False
            
            parts = []
            if count_paragraphs(orig) != count_paragraphs(trans):
                parts.append("Mismatched number of paragraphs")
            if first_char_type(orig) != first_char_type(trans):
                parts.append("Different first character type")
            if last_char_type(orig) != last_char_type(trans):
                parts.append("Different last character type")
            if extract_placeholders(orig) != extract_placeholders(trans):
                parts.append("Mismatched placeholders")
            if orig and trans and (abs(len(orig) - len(trans)) > 0.5 * max(len(orig), len(trans))):
                parts.append("Significant length difference")
            if extract_numbers(orig) != extract_numbers(trans):
                parts.append("Mismatched numbers")
            if extract_formatting(orig) != extract_formatting(trans):
                parts.append("Mismatched formatting (bold/italic/code)")
            if extract_urls(orig) != extract_urls(trans):
                parts.append("Mismatched URLs or links")
            if count_brackets_quotes(orig) != count_brackets_quotes(trans):
                parts.append("Mismatched brackets or quotes")
            if abs(count_sentence_starts(orig) - count_sentence_starts(trans)) > 1:
                parts.append("Different sentence capitalization pattern")
            if extract_special_chars(orig) != extract_special_chars(trans):
                parts.append("Mismatched special characters or emojis")
            if has_list_structure(orig) != has_list_structure(trans):
                parts.append("Mismatched list structure")
            
            item.setToolTip("Translation issues:\n- " + "\n- ".join(parts))
        else:
            item.setToolTip("")

    def display_selected_fragment(self, current_item, previous_item):
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        self.original_text_view.setText(self.paragraphs[idx]['original_text'])
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.setText(self.paragraphs[idx]['translated_text'])
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

    def update_translation_from_edit(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        edited_text = self.translated_text_view.toPlainText()
        self.paragraphs[idx]['translated_text'] = edited_text
        if edited_text and not self.paragraphs[idx]['is_translated']:
            self.paragraphs[idx]['is_translated'] = True
            self.update_item_visuals(current_item, self.paragraphs[idx])

    def check_mismatch(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            if self.paragraphs[idx]['is_translated']:
                self.update_item_visuals(item, self.paragraphs[idx])

    def start_auto_fix_process(self):
        to_retry = []
        for idx in self.selected_for_auto_fix.copy():
            if self._has_mismatch(idx):
                if self.auto_fix_attempts.get(idx, 0) < self.max_auto_fix_attempts:
                    to_retry.append(idx)
                    self.auto_fix_attempts[idx] = self.auto_fix_attempts.get(idx, 0) + 1
                else:
                    self.selected_for_auto_fix.discard(idx)
            else:
                self.selected_for_auto_fix.discard(idx)
        if to_retry:
            self.statusBar().showMessage(f"Auto-fix: retrying translation for {len(to_retry)} fragments...", 0)
            if not hasattr(self, 'retry_workers'):
                self.retry_workers = []
            for idx in to_retry:
                self.retry_paragraph(idx)
        else:
            self.finalize_translation()

    def retry_paragraph(self, idx: int):
        if not hasattr(self, 'retry_workers'):
            self.retry_workers = []
        original = self.paragraphs[idx]['original_text']
        retry_temp = min(self.temperature_spinbox.value() + 0.1, 1.0)
        llm_choice = self.app_settings.get("llm_choice", "LM Studio")
        model_name = self.app_settings.get("ollama_model_name", "") if llm_choice == "Ollama" else \
                     self.app_settings.get("openrouter_model_name", "") if llm_choice == "Openrouter" else "local-model"
        openrouter_api_key = self.app_settings.get("openrouter_api_key", "") if llm_choice == "Openrouter" else None
        worker = TranslationWorker(
            paragraphs_to_translate=[(idx, original)],
            llm_instruction=self.llm_system_prompt.toPlainText(),
            context_size=self.context_spinbox.value(),
            temperature=retry_temp,
            all_paragraphs=self.paragraphs,
            llm_choice=llm_choice,
            model_name=model_name,
            openrouter_api_key=openrouter_api_key,
            custom_ollama_prompt=self.custom_ollama_prompt,
            custom_system_prompt=self.custom_system_prompt,
            custom_user_prompt=self.custom_user_prompt
        )
        worker.progress.connect(self.on_retry_progress)
        def _cleanup():
            try:
                self.retry_workers.remove(worker)
            except ValueError:
                pass
            worker.deleteLater()
            self._check_auto_fix_complete()
        worker.finished.connect(_cleanup)
        self.retry_workers.append(worker)
        worker.start()

    def on_retry_progress(self, idx, translated_text, is_error):
        self.paragraphs[idx]['translated_text'] = translated_text
        self.paragraphs[idx]['is_translated'] = not is_error
        item = self.list_widget.item(idx)
        if item:
            self.update_item_visuals(item, self.paragraphs[idx])
        if self.list_widget.currentItem() == item:
            self.display_selected_fragment(item, None)

    def _check_auto_fix_complete(self):
        active_workers = [w for w in getattr(self, 'retry_workers', []) if w.isRunning()]
        if not active_workers:
            if getattr(self, 'auto_fix_checkbox', None) and self.auto_fix_checkbox.isChecked():
                remaining_mismatch = [idx for idx in self.selected_for_auto_fix if self._has_mismatch(idx)]
                if remaining_mismatch:
                    self.start_auto_fix_process()
                else:
                    self.finalize_translation()
            else:
                self.finalize_translation()

    def start_translation(self):
        selected_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                selected_items.append((idx, self.paragraphs[idx]['original_text']))
        if not selected_items:
            self.show_message("No Selection", "Select at least one fragment to translate.", QMessageBox.Icon.Warning)
            return
        self.total_to_translate = len(selected_items)
        self.completed_translations = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.selected_for_auto_fix = {idx for idx, _ in selected_items}
        self.max_auto_fix_attempts = self.auto_fix_spinbox.value()
        self.auto_fix_attempts = {idx: 0 for idx in self.selected_for_auto_fix}
        system_prompt = self.llm_system_prompt.toPlainText()
        temp_value = self.temperature_spinbox.value()
        llm_choice = self.app_settings.get("llm_choice", "LM Studio")
        model_name = self.app_settings.get("ollama_model_name", "") if llm_choice == "Ollama" else \
                     self.app_settings.get("openrouter_model_name", "") if llm_choice == "Openrouter" else "local-model"
        openrouter_api_key = self.app_settings.get("openrouter_api_key", "") if llm_choice == "Openrouter" else None
        if llm_choice == "Ollama" and not model_name:
            self.show_message("Missing Model", "For Ollama, you must set the model name (e.g., llama3.2:3b)", QMessageBox.Icon.Warning)
            return
        elif llm_choice == "Openrouter" and (not openrouter_api_key or not model_name):
            self.show_message("Missing Settings", "For Openrouter, you must provide API key and model name.", QMessageBox.Icon.Warning)
            return
        self.translation_worker = TranslationWorker(
            paragraphs_to_translate=selected_items,
            llm_instruction=system_prompt,
            context_size=self.context_spinbox.value(),
            temperature=temp_value,
            all_paragraphs=self.paragraphs,
            llm_choice=llm_choice,
            model_name=model_name,
            openrouter_api_key=openrouter_api_key,
            custom_ollama_prompt=self.custom_ollama_prompt,
            custom_system_prompt=self.custom_system_prompt,
            custom_user_prompt=self.custom_user_prompt
        )
        self.translation_worker.progress.connect(self.on_translation_progress)
        self.translation_worker.finished.connect(self.on_translation_finished)
        self.statusBar().showMessage("Translation started...", 0)
        self.translation_worker.start()

    def on_translation_progress(self, idx, translated_text, is_error):
        self.paragraphs[idx]['translated_text'] = translated_text
        self.paragraphs[idx]['is_translated'] = not is_error
        item = self.list_widget.item(idx)
        if item:
            self.update_item_visuals(item, self.paragraphs[idx])
            item.setCheckState(Qt.CheckState.Unchecked)
        if self.list_widget.currentItem() == item:
            self.display_selected_fragment(item, None)
        self.completed_translations += 1
        percent = int(self.completed_translations / self.total_to_translate * 100)
        self.progress_bar.setValue(percent)

    def on_translation_finished(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            self.update_item_visuals(item, self.paragraphs[i])
        self.progress_bar.setVisible(False)
        if getattr(self, 'auto_fix_checkbox', None) and self.auto_fix_checkbox.isChecked():
            self.start_auto_fix_process()
        else:
            self.finalize_translation()

    def finalize_translation(self):
        self.statusBar().showMessage("Translation completed.", 5000)
        remaining_mismatch = []
        if hasattr(self, 'selected_for_auto_fix'):
            remaining_mismatch = [idx for idx in self.selected_for_auto_fix if self._has_mismatch(idx)]
        if remaining_mismatch:
            self.show_message(
                "Translation Completed with Warnings", 
                f"Translation completed.\n\nNote: {len(remaining_mismatch)} fragments still have mismatches.\nYou can try translating them again manually or check LLM settings.",
                QMessageBox.Icon.Warning
            )
        else:
            self.show_message("Completed", "Translation of selected fragments is complete.")
        if hasattr(self, 'selected_for_auto_fix'):
            self.selected_for_auto_fix.clear()
        if hasattr(self, 'auto_fix_attempts'):
            self.auto_fix_attempts.clear()

    def save_file(self):
        if not self.paragraphs:
            self.show_message("No Data", "First, open a file.", QMessageBox.Icon.Warning)
            return
        if self.file_type == "epub":
            path, _ = QFileDialog.getSaveFileName(self, "Save as New EPUB", "", "EPUB Files (*.epub)")
            if not path:
                return
            self.epub_creator = EPUBCreator(self.book, self.paragraphs, path)
            self.epub_creator.finished.connect(self.on_file_saved)
            self.epub_creator.start()
        elif self.file_type == "srt":
            path, _ = QFileDialog.getSaveFileName(self, "Save as New SRT", "", "SRT Files (*.srt)")
            if not path:
                return
            self.srt_creator = SRTCreator(self.paragraphs, path)
            self.srt_creator.finished.connect(self.on_file_saved)
            self.srt_creator.start()
        self.statusBar().showMessage("Saving file...")

    def on_file_saved(self, path, is_error):
        if is_error:
            self.show_message("Save Error", f"Failed to save file:\n{path}", QMessageBox.Icon.Critical)
        else:
            self.show_message("Success", f"File saved:\n{path}")

    def save_session(self):
        if not self.paragraphs:
            self.show_message("No Data", "No progress to save.", QMessageBox.Icon.Warning)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON Files (*.json)")
        if not path:
            return
        session_data = {
            'original_file_path': self.original_file_path,
            'file_type': self.file_type,
            'paragraphs': self.paragraphs,
            'system_prompt': self.llm_system_prompt.toPlainText(),
            'context_size': self.context_spinbox.value(),
            'temperature': self.temperature_spinbox.value(),
            'custom_ollama_prompt': self.custom_ollama_prompt,
            'custom_system_prompt': self.custom_system_prompt,
            'custom_user_prompt': self.custom_user_prompt
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4)
            self.show_message("Success", f"Session saved to file:\n{path}")
        except Exception as e:
            self.show_message("Session Save Error", f"Failed to save session:\n{e}", QMessageBox.Icon.Critical)

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            original_path = session_data.get('original_file_path')
            if not original_path:
                self.show_message("Error", "No original file path in session.", QMessageBox.Icon.Critical)
                return
            confirmed_path, _ = QFileDialog.getOpenFileName(
                self, "Confirm original file location", original_path, "Files (*.epub *.srt)"
            )
            if not confirmed_path:
                self.show_message("Error", "No original file selected.", QMessageBox.Icon.Critical)
                return
            self.file_type = session_data.get('file_type', 'epub')
            if self.file_type == "epub":
                self.open_epub_with_session(confirmed_path, session_data['paragraphs'])
            elif self.file_type == "srt":
                self.paragraphs = session_data['paragraphs']
                self.original_file_path = confirmed_path
                self.populate_list()
                self.show_message("Success", "Session loaded successfully.")
            if 'system_prompt' in session_data:
                self.llm_system_prompt.setPlainText(session_data['system_prompt'])
            if 'context_size' in session_data:
                self.context_spinbox.setValue(session_data['context_size'])
            if 'temperature' in session_data:
                self.temperature_spinbox.setValue(session_data['temperature'])
            self.custom_ollama_prompt = session_data.get('custom_ollama_prompt')
            self.custom_system_prompt = session_data.get('custom_system_prompt')  
            self.custom_user_prompt = session_data.get('custom_user_prompt')
        except Exception as e:
            self.show_message("Session Load Error", f"Failed to load session file:\n{e}", QMessageBox.Icon.Critical)

    def open_epub_with_session(self, epub_path, session_paragraphs):
        """
        Load an EPUB, re-insert saved fragment IDs, and restore original_html and translation state from session data.
        """
        self.original_file_path = epub_path
        try:
            # Read EPUB
            self.book = epub.read_epub(epub_path)

            # Define which tags to process (block-level)
            TAGS_TO_EXTRACT = [
                "h1","h2","h3","h4","h5","h6",
                "p","li","td","th","blockquote","pre"
            ]

            # Build lookup: by (href, original_text) -> session info
            session_map = {
                (p['item_href'], p['original_text']): p
                for p in session_paragraphs
            }

            # Iterate document items and re-insert IDs
            for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                raw = item.get_content()
                if not raw:
                    continue
                html = raw.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')

                for tag_name in TAGS_TO_EXTRACT:
                    for elem in soup.find_all(tag_name):
                        text = elem.get_text(separator=" ", strip=True)
                        key = (item.get_name(), text)
                        if key not in session_map:
                            continue

                        info = session_map[key]
                        # Ensure ID attribute
                        elem['id'] = info['id']

                # Save updated content back to book
                item.set_content(str(soup).encode('utf-8'))

            # Restore paragraphs list with full session data
            # Make a shallow copy to avoid mutating external list
            self.paragraphs = [dict(p) for p in session_paragraphs]

            self.populate_list()
            self.show_message("Success", "Progress loaded from session file.")

        except Exception as e:
            self.show_message(
                "EPUB Load Error",
                f"Failed to load EPUB file:\n{e}",
                QMessageBox.Icon.Critical
            )

    def toggle_all_selection(self, check):
        state = Qt.CheckState.Checked if check else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def filter_list(self, show_translated):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            is_translated = self.paragraphs[idx]['is_translated']
            if show_translated is None:
                item.setHidden(False)
            else:
                item.setHidden(is_translated != show_translated)

    def filter_mismatch(self, show_mismatch: bool):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            mismatch = self._has_mismatch(i)
            item.setHidden(mismatch != show_mismatch)

    def cancel_translation(self):
        if hasattr(self, 'translation_worker') and self.translation_worker.isRunning():
            self.translation_worker.terminate()
            self.translation_worker.wait()
        if hasattr(self, 'retry_workers'):
            for w in list(self.retry_workers):
                if w.isRunning():
                    w.terminate()
                    w.wait()
            self.retry_workers.clear()
        if hasattr(self, 'selected_for_auto_fix'):
            self.selected_for_auto_fix.clear()
        if hasattr(self, 'auto_fix_attempts'):
            self.auto_fix_attempts.clear()
        self.statusBar().showMessage("Translation cancelled.", 5000)

    def closeEvent(self, event):
        self.cancel_translation()
        if hasattr(self, 'epub_creator') and self.epub_creator.isRunning():
            self.epub_creator.terminate()
            self.epub_creator.wait(5000)
        if hasattr(self, 'srt_creator') and self.srt_creator.isRunning():
            self.srt_creator.terminate()
            self.srt_creator.wait(5000)
        if hasattr(self, 'retry_workers'):
            for w in list(self.retry_workers):
                if w.isRunning():
                    w.terminate()
                    w.wait(5000)
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TranslatorApp()
    ex.show()
    sys.exit(app.exec())