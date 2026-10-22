"""Check local documentation targets, Python syntax, docstrings and HTML API coverage.

Run from the repository root with Python; only the standard library is required.
Reads maintained Python/Markdown and docs/html without importing project modules.
Prints JSON evidence; checks local HTML href/src targets and fragment IDs, plus
inline Markdown file links. External URLs, JavaScript behaviour, CSS url() and
Markdown reference-style links are outside this check. Returns nonzero on failure.
"""
import ast
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    """Collect HTML identifiers and local hyperlink/asset references."""

    def __init__(self, text):
        """Parse UTF-8 HTML text into ids and href/src collections.

        Args:
            text: str; complete generated HTML document.
        """
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        """Collect id/name anchors and href/src attributes from one element.

        Args:
            tag: str; HTML tag name supplied by HTMLParser.
            attrs: list[tuple[str, str | None]]; parsed attributes.
        """
        for key, value in attrs:
            if value and (key == 'id' or key == 'name' and tag == 'a'):
                self.ids.add(value)
            if value and key in ('href', 'src'):
                self.links.append(value)


def check():
    """Compile maintained Python and resolve local Markdown/HTML references.

    Returns:
        dict: Inventories, coverage counts and explicit failure lists. No files
        are written; Python compilation occurs in memory without execution.

    Raises:
        OSError: If maintained source or documentation cannot be read.
        SyntaxError: If a maintained Python file does not compile or parse.
    """
    python = sorted(p for p in ROOT.rglob('*.py')
                    if not any(x in p.parts for x in ('.git', '_build', '__pycache__')))
    result = {'python': [], 'broken_links': [], 'missing_api': [],
              'missing_api_members': [], 'rendered_api_definitions': 0,
              'missing_parameter_docs': [], 'checked_parameters': 0,
              'missing_rendered_parameters': [], 'rendered_parameter_fields': 0}
    for path in python:
        source = path.read_text(encoding='utf-8-sig')
        compile(source, str(path), 'exec')
        tree = ast.parse(source)
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parameters = [a.arg for a in [*node.args.posonlyargs, *node.args.args,
                          *node.args.kwonlyargs] if a.arg not in ('self', 'cls')]
            doc = ast.get_docstring(node) or ''
            section = doc.split('Args:', 1)[1] if 'Args:' in doc else ''
            section = re.split(r'\n(?:Returns|Yields|Raises|Note|Notes):', section)[0]
            documented = set(re.findall(r'^\s*(\w+)(?:\s*\([^\n]*\))?:', section, re.M))
            result['checked_parameters'] += len(parameters)
            for name in parameters:
                if name not in documented:
                    result['missing_parameter_docs'].append(
                        f'{path.relative_to(ROOT).as_posix()}:{node.lineno}:{node.name}:{name}')
        result['python'].append(dict(path=path.relative_to(ROOT).as_posix(),
            functions=sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes),
            classes=sum(isinstance(n, ast.ClassDef) for n in nodes),
            missing_module_docstring=not bool(ast.get_docstring(tree)),
            missing_docstrings=[n.name for n in nodes if not ast.get_docstring(n)]))
    pages = {p.resolve(): Page(p.read_text(encoding='utf-8'))
             for p in (ROOT / 'docs/html').rglob('*.html')}
    html_links = 0
    for path, page in pages.items():
        for target in page.links:
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            html_links += 1
            destination = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not destination.exists() or (url.fragment and destination in pages
                    and unquote(url.fragment) not in pages[destination].ids):
                result['broken_links'].append([path.relative_to(ROOT).as_posix(), target])
    markdown = sorted(p for p in ROOT.rglob('*.md')
                      if not any(x in p.parts for x in ('.git', '_build', 'html'))
                      and (p.parts[-2] != 'outputs' or p.name == 'README.md')
                      and ('outputs' not in p.relative_to(ROOT).parts or p.name == 'README.md'))
    markdown_links = 0
    for path in markdown:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8-sig')):
            target = target.strip('<>')
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            markdown_links += 1
            if not (path.parent / unquote(url.path)).exists():
                result['broken_links'].append([path.relative_to(ROOT).as_posix(), target])
    source = sorted((ROOT / 'src').glob('*.py'))
    api_index = (ROOT / 'docs/html/api/index.html').read_text(encoding='utf-8')
    for path in source:
        page = ROOT / 'docs/html/api' / (path.stem + '.html')
        if not page.exists() or path.stem + '.html' not in api_index:
            result['missing_api'].append(path.name)
        else:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            rst = (ROOT / 'docs/source/api' / (path.stem + '.rst')).read_text(encoding='utf-8')
            blocks = re.split(r'(?=^\.\. py:)', rst, flags=re.M)
            fields = {}
            for block in blocks:
                signature = re.match(r'\.\. py:(?:function|method):: ([^(]+)\(', block)
                if signature:
                    fields[signature[1]] = set(re.findall(r':param (\w+):', block))
            pending = [(tree, path.stem)]
            while pending:
                node, prefix = pending.pop()
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = prefix + '.' + child.name
                        if name not in pages[page.resolve()].ids:
                            result['missing_api_members'].append(name)
                        else:
                            result['rendered_api_definitions'] += 1
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for argument in [*child.args.posonlyargs, *child.args.args,
                                             *child.args.kwonlyargs]:
                                if argument.arg in ('self', 'cls'):
                                    continue
                                if argument.arg not in fields.get(name[len(path.stem) + 1:], set()):
                                    result['missing_rendered_parameters'].append(name + ':' + argument.arg)
                                else:
                                    result['rendered_parameter_fields'] += 1
                        pending.append((child, name))
                    else:
                        pending.append((child, prefix))
    result.update(compiled_modules=len(python), production_modules=len(source),
                  html_pages=len(pages), html_local_references=html_links,
                  markdown_files=len(markdown), markdown_local_file_links=markdown_links)
    return result


if __name__ == '__main__':
    evidence = check()
    print(json.dumps(evidence, indent=2))
    sys.exit(bool(evidence['broken_links'] or evidence['missing_api'] or
                  evidence['missing_api_members'] or evidence['missing_parameter_docs'] or
                  evidence['missing_rendered_parameters'] or any(
        row['missing_docstrings'] or row['missing_module_docstring'] for row in evidence['python'])))
