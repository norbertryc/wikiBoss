import re
import mwparserfromhell

from .utils import save_in_batches, track_progress_and_time, DataLoader
from .logging import logger


class Cleaner(DataLoader):
    """"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {"tables_tree": 0, "tables_raw": 0, "tables_failed": 0}

    def _extract_metadata(self, wikicode: mwparserfromhell.wikicode.Wikicode
                          ) -> dict:
        """"""
        metadata = {}

        # categories
        categories = []
        for link in wikicode.filter_wikilinks(matches="Kategoria"):
            full = str(link.title).strip()
            category = full.split(":")[-1]
            categories.append(category) if category else None
        metadata["categories"] = categories

        # maybe other metadata extraction

        return metadata

    def _remove_trash_sections(self, wikicode: mwparserfromhell.wikicode.Wikicode
                                ) -> mwparserfromhell.wikicode.Wikicode:
        """Remove trash nodes and wikilinks."""
        trash_headers = ["Przypisy", "Zobacz też", "Linki zewnętrzne", "Kategoria", "Bibliografia", "Uwagi"]
        nodes = wikicode.nodes
        to_remove = set()
        remove_from = None

        for i, node in enumerate(nodes):
            if isinstance(node, mwparserfromhell.nodes.heading.Heading):
                heading_text = str(node.title).strip()
                if heading_text in trash_headers:
                    remove_from = i

            if remove_from is not None:
                to_remove.add(i)

        for i in sorted(to_remove, reverse=True):
            wikicode.remove(nodes[i])

        for wikilink in wikicode.filter_wikilinks():
            if str(wikilink.title).lower().startswith(("plik:", "kategoria:")):
                try:
                    wikicode.remove(wikilink, recursive=True)
                except Exception as e:
                    logger.warning(f"Wikilink can't be removed because of exception: '{e}'")

        return wikicode

    def _convert_headings_to_markdown(self, wikicode: mwparserfromhell.wikicode.Wikicode
                                      ) -> mwparserfromhell.wikicode.Wikicode:
        """Convert wikimedia headings style to markdown"""
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

    def _convert_math_nodes_to_latex(self, wikicode: mwparserfromhell.wikicode.Wikicode
                            ) -> mwparserfromhell.wikicode.Wikicode:
        """
        Convert 'math' tags to save their content (mwparserfromhell's strip_code() method removes
        'math' tags with content by default)
        """
        new_nodes = []

        for node in wikicode.nodes:
            if isinstance(node, mwparserfromhell.nodes.tag.Tag) and node.tag == "math":
                content = node.contents.strip_code().strip()
                latex_block = f"$${content}$$"
                new_nodes.append(mwparserfromhell.nodes.text.Text(latex_block))
            else:
                new_nodes.append(node)
        new_wikicode = mwparserfromhell.wikicode.Wikicode(new_nodes)

        return new_wikicode

    def _convert_special_templates(self, wikicode: mwparserfromhell.wikicode.Wikicode
                                   ) -> mwparserfromhell.wikicode.Wikicode:
        """"""
        def get_arg(template, n):
            if not template.has(n):
                return ""
            return template.get(n).value.strip_code().strip()

        handlers = {
            "k": lambda t: get_arg(t, 2),
            "j": lambda t: get_arg(t, 2),
            "klawisz": lambda t: get_arg(t, 1),
            "ang.": lambda t: get_arg(t, 1),
        }

        for template in wikicode.filter_templates():
            name = template.name.strip_code().strip().lower()
            handler = handlers.get(name)

            if handler:
                try:
                    wikicode.replace(template, handler(template))
                except Exception as e:
                    logger.warning(f"Error processing template '{name}': {e}")

        return wikicode

    def _convert_tables_to_md(self, wikicode: mwparserfromhell.wikicode.Wikicode
                              ) -> mwparserfromhell.wikicode.Wikicode:
        """Convert mediawiki tables to markdown."""
        table_pattern = re.compile(r'\{\|(.*?)\|\}', re.DOTALL)

        for node in wikicode.filter_tags(matches=lambda n: n.tag == "table", recursive=False):
            try:
                # check content structure
                content_str = str(node.contents).strip()
                is_raw_text = content_str.startswith('{|') or '!!' in content_str[:10]

                markdown = ""
                if is_raw_text:
                    # table tag containing raw wikitext
                    markdown = self._parse_raw_wikitable_to_md(content_str)
                    if markdown:
                        self.stats["tables_raw"] += 1
                else:
                    # tree table
                    markdown = self._parse_table_tag_to_md(node)
                    if markdown:
                        self.stats["tables_tree"] += 1

                if markdown:
                    wikicode.replace(node, mwparserfromhell.nodes.text.Text(markdown))

            except Exception as e:
                self.stats["tables_failed"] += 1
                logger.warning(f"Table tag conversion failed: {e}")

        # scan remaining text for nested tables
        for node in wikicode.filter_text():
            if '{|' in str(node):
                text_content = str(node)
                if table_pattern.search(text_content):
                    new_content = text_content
                    for match in table_pattern.finditer(text_content):
                        full_match = match.group(0)
                        try:
                            md_table = self._parse_raw_wikitable_to_md(full_match)
                            if md_table:
                                new_content = new_content.replace(full_match, md_table)
                                self.stats["tables_raw"] += 1
                        except Exception as e:
                            logger.debug(f"Raw table conversion failed: {e}")

                    if new_content != text_content:
                        wikicode.replace(node, mwparserfromhell.nodes.text.Text(new_content))

        return wikicode

    def _parse_table_tag_to_md(self, table_node: mwparserfromhell.nodes.tag.Tag) -> str:
        """Convert a mwparserfromhell Tag object (table) to markdown using Tree structure."""
        rows = []

        tr_nodes = table_node.contents.filter_tags(matches=lambda n: n.tag == "tr", recursive=False)
        if not tr_nodes:
            tr_nodes = table_node.contents.filter_tags(matches=lambda n: n.tag == "tr", recursive=True)
        if not tr_nodes:
            return ""

        for row in tr_nodes:
            cells = []
            is_header_row = False

            cell_nodes = row.contents.filter_tags(matches=lambda n: n.tag in ("td", "th"), recursive=False)

            for cell in cell_nodes:
                if cell.tag == "th":
                    is_header_row = True

                content = cell.contents.strip_code(collapse=True).strip()
                content = content.replace("|", "&#124;").replace("\n", " ")
                cells.append(content)

            if cells:
                rows.append((cells, is_header_row))

        return self._build_md_table_from_rows(rows)

    def _parse_raw_wikitable_to_md(self, raw_table_text: str) -> str:
        """Convert raw text tables to markdown."""
        lines = raw_table_text.strip().split('\n')
        if len(lines) < 2:
            return ""

        start_idx = 1 if lines[0].startswith('{|') else 0
        end_idx = -1 if lines[-1].startswith('|}') else len(lines)
        processing_lines = lines[start_idx:end_idx]

        rows = []
        current_row_cells = []
        is_header_row = False

        for line in processing_lines:
            line = line.strip()
            if not line:
                continue

            # new row
            if line.startswith('|-') or line.startswith('| -'):
                if current_row_cells:
                    rows.append((current_row_cells, is_header_row))
                    current_row_cells = []
                    is_header_row = False
                continue

            # cell
            if line.startswith('|') or line.startswith('!'):
                is_header = line.startswith('!')
                if is_header:
                    is_header_row = True

                content = line[1:]
                if '|' in content:
                    parts = content.split('|', 1)
                    html_attr = parts[0].lower()
                    if any(attr in html_attr for attr in ['style=', 'class=', 'align=', 'width=', 'colspan=']):
                        content = parts[1]

                separator = '!!' if is_header else '||'
                sub_cells = content.split(separator)

                for sub_cell in sub_cells:
                    clean_content = self._clean_raw_table_cell(sub_cell)
                    if clean_content:
                        current_row_cells.append(clean_content)

        if current_row_cells:
            rows.append((current_row_cells, is_header_row))

        return self._build_md_table_from_rows(rows)

    def _clean_raw_table_cell(self, text: str) -> str:
        """Clean raw cell text."""
        # clean wikilinks
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        # remove style attributes
        text = re.sub(r'style="[^"]*"', '', text)
        # remove bold/italic
        text = text.replace("'''", "").replace("''", "")
        # escape pipes
        text = text.replace("|", "&#124;")

        return text.strip()

    def _build_md_table_from_rows(self, rows: list) -> str:
        """Build markdown table string from parsed rows."""
        if not rows:
            return ""

        header_row = None
        data_rows = []

        for cells, is_header in rows:
            if header_row is None and (is_header or not header_row):
                header_row = cells
            else:
                data_rows.append(cells)

        if header_row is None:
            # take first row as header if no header
            if data_rows:
                header_row = data_rows.pop(0)
            else:
                return ""

        num_cols = len(header_row)
        for row in data_rows:
            if len(row) > num_cols:
                num_cols = len(row)

        def normalize(r):
            return "| " + " | ".join(r + [""] * (num_cols - len(r))) + " |"

        md_lines = []
        md_lines.append(normalize(header_row))
        md_lines.append("|" + "|".join(["---"] * num_cols) + "|")
        for row in data_rows:
            md_lines.append(normalize(row))

        return "\n".join(md_lines) + "\n\n"

    def _strip_whitespace(self, text: str) -> str:
        """"""
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    @save_in_batches()
    @track_progress_and_time("Cleaning")
    def clean(self):
        """"""
        for i, article in enumerate(self.articles):
            try:
                wikicode = mwparserfromhell.parse(article["text"])
                article.update( self._extract_metadata(wikicode))

                wikicode = self._remove_trash_sections(wikicode)
                wikicode = self._convert_headings_to_markdown(wikicode)
                wikicode = self._convert_math_nodes_to_latex(wikicode)
                wikicode = self._convert_special_templates(wikicode)
                wikicode = self._convert_tables_to_md(wikicode)

                text = wikicode.strip_code(collapse=True)
                text = self._strip_whitespace(text)

                article["text"] = text

                yield article

            except Exception as e:
                logger.error(f"The article no {i} (wiki id {article["id"]}): {article["title"]} skipped"
                             f" because of exception:\n'{e}'")
