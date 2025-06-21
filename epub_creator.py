import logging
from PyQt6.QtCore import QThread, pyqtSignal
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag
import uuid
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EPUBCreator(QThread):
    finished = pyqtSignal(str, bool)

    def __init__(self, book, paragraphs, output_path):
        super().__init__()
        self.book = book
        self.paragraphs = paragraphs
        self.output_path = output_path

    def run(self):
        try:
            logger.debug(f"Starting EPUB save to: {self.output_path}")

            # 1) Zbierz oryginalne nagłówki XML/DOCTYPE
            headers = {}
            for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                raw = item.get_content()
                if not raw:
                    continue
                content = raw.decode('utf-8', errors='ignore')
                prefix = ''
                if content.lstrip().startswith('<?xml'):
                    parts = content.split('?>', 1)
                    prefix = parts[0] + '?>\n'
                    content = parts[1]
                doctype = ''
                if '<!DOCTYPE' in content:
                    head, tail = content.split('<!DOCTYPE', 1)
                    decl = '<!DOCTYPE' + tail.split('>', 1)[0] + '>\n'
                    doctype = decl
                    content = head + content[len(head) + len(decl):]
                headers[item.get_name()] = (prefix, doctype)

            # 2) Wstaw przetłumaczenia, zachowując oryginalną strukturę HTML
            for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                raw = item.get_content()
                if not raw:
                    continue
                html = raw.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')

                for p in self.paragraphs:
                    if not p.get('is_translated') or p['item_href'] != item.get_name():
                        continue

                    elem = soup.find(p['element_type'], attrs={'id': p['id']})
                    if not elem:
                        continue

                    # Utwórz kopię oryginalnej struktury HTML
                    original_soup = BeautifulSoup(p['original_html'], 'html.parser')
                    original_elem = original_soup.find(p['element_type'])

                    # — nowa logika: jeśli w oryginale jest span.calibre1, podstaw w nim tekst —
                    title_span = original_elem.find('span', class_='calibre1')
                    if title_span:
                        # usuń powtórzone numerowanie, jeśli jest obok <span class="item-number">
                        parent_p = original_elem
                        num_span = parent_p.find('span', class_='item-number')
                        new_text = p['translated_text']
                        if num_span:
                            # obetnij wiodące "1. " lub "12. "
                            new_text = re.sub(r'^\s*\d+\.\s*', '', new_text)
                        title_span.string = new_text
                    else:
                        # dotychczasowa logika dla zwykłych fragmentów tekstu
                        text_nodes = [
                            node for node in original_elem.descendants
                            if isinstance(node, NavigableString) and node.strip()
                        ]
                        if text_nodes:
                            if len(text_nodes) == 1:
                                text_nodes[0].replace_with(p['translated_text'])
                            else:
                                main = max(text_nodes, key=lambda x: len(x))
                                main.replace_with(p['translated_text'])
                                for node in text_nodes:
                                    if node is not main:
                                        node.replace_with('')

                    # Zastąp element w dokumencie zmodyfikowaną wersją
                    elem.replace_with(original_elem)

                # 3) Odtwórz nagłówki i zapisz zmienioną treść
                prefix, doctype = headers.get(item.get_name(), ('', ''))
                new_html = prefix + doctype + str(soup)
                item.set_content(new_html.encode('utf-8'))

            # 4) Zapisz nowy EPUB
            epub.write_epub(self.output_path, self.book)
            logger.debug("EPUB save completed successfully")
            self.finished.emit(self.output_path, False)

        except Exception as e:
            logger.exception("Error during EPUB creation")
            self.finished.emit(str(e), True)