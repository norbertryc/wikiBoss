import re
import mwparserfromhell

from .utils import save_in_batches, track_progress_and_time, DataLoader
from .logging import logger


class Cleaner(DataLoader):

    def _extract_metadata(self, text: str):
        """"""

    def _remove_thrash_sections(self, wikicode: mwparserfromhell.wikicode.Wikicode):
        """"""
        thrash_headers = ["Przypisy", "Zobacz też", "Linki zewnętrzne", "Kategoria", "Bibliografia", "Uwagi"]
        nodes = wikicode.nodes
        to_remove = set()
        remove_from = None

        for i, node in enumerate(nodes):
            if isinstance(node, mwparserfromhell.nodes.heading.Heading):
                heading_text = str(node.title).strip()
                if heading_text in thrash_headers:
                    remove_from = i

            if remove_from is not None:
                to_remove.add(i)

        for i in sorted(to_remove, reverse=True):
            wikicode.remove(nodes[i])

        for wikilink in wikicode.filter_wikilinks():
            if str(wikilink.title).lower().startswith(("plik:", "kategoria:")):
                wikicode.remove(wikilink)
        return wikicode

    def _convert_headings_to_markdown(self, wikicode: mwparserfromhell.wikicode.Wikicode):
        """"""
        new_nodes = []

        for node in wikicode.nodes:
            if isinstance(node, mwparserfromhell.nodes.heading.Heading):
                level = node.level
                header_text = str(node.title).strip()
                md = "#" * level + f" {header_text}\n"
                new_nodes.append(mwparserfromhell.nodes.text.Text(md))
            else:
                new_nodes.append(node)
        new_wikicode = mwparserfromhell.wikicode.Wikicode(new_nodes)
        return new_wikicode

    def _strip_whitespace(self, text: str):
        """"""
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    @save_in_batches()
    @track_progress_and_time("Basic cleaning")
    def clean_basic(self):
        """"""
        for article in self.articles:

            wikicode = mwparserfromhell.parse(article["text"])
            wikicode = self._remove_thrash_sections(wikicode)
            wikicode = self._convert_headings_to_markdown(wikicode)

            text = wikicode.strip_code()
            text = self._strip_whitespace(text)

            article["text"] = text

            yield article

    @save_in_batches()
    @track_progress_and_time("Advanced cleaning")
    def clean_advanced(self):
        """"""

