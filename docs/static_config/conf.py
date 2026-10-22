"""Static API extraction: no project modules are imported or executed."""
import ast
from pathlib import Path
import runpy
from sphinx.ext.napoleon.docstring import GoogleDocstring

ROOT = Path(__file__).resolve().parents[2]
globals().update({k: v for k, v in runpy.run_path(str(ROOT / 'docs/source/conf.py')).items() if not k.startswith('__')})
project = 'Apple Tree Localiser — MMA3001'
author = 'Chan Lik Wai'
copyright = '2026, Chan Lik Wai'
html_title = 'MMA3001 Source Documentation'
html_theme_options = {'page_width': '1200px', 'sidebar_width': '280px'}
api = ROOT / 'docs/source/api'
api.mkdir(exist_ok=True)

def prose(node):
    """Format a parsed object docstring without importing its module.

    Args:
        node: ast.AST module, class or function supporting ast.get_docstring.

    Returns:
        str: Napoleon-formatted reStructuredText, or a missing-docstring label.
    """
    doc = ast.get_docstring(node)
    return str(GoogleDocstring(doc)) if doc else 'No docstring is present in the source.'

def members(nodes, prefix=''):
    """Recursively emit static Python-domain directives, including nested helpers.

    Args:
        nodes: Iterable[ast.AST] statements from a parsed body.
        prefix: str; enclosing qualified name and trailing dot, or empty.

    Returns:
        list[str]: RST lines with signatures and formatted documentation.
    """
    lines = []
    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            if isinstance(node, ast.ClassDef):
                kind = 'class'
                bases = ', '.join(ast.unparse(b) for b in node.bases)
                signature = '(' + bases + ')' if bases else ''
            else:
                kind = 'method' if prefix else 'function'
                signature = '(' + ast.unparse(node.args) + ')'
                if node.returns:
                    signature += ' -> ' + ast.unparse(node.returns)
            lines += [f'.. py:{kind}:: {name}{signature}', '']
            lines += ['   ' + line if line.strip() else '' for line in prose(node).splitlines()] + ['']
            lines += members(node.body, name + '.')
        else:
            for field in ('body', 'orelse', 'finalbody'):
                child = getattr(node, field, None)
                if isinstance(child, list):
                    lines += members(child, prefix)
            for handler in getattr(node, 'handlers', []):
                lines += members(handler.body, prefix)
    return lines

names = []
for path in sorted((ROOT / 'src').rglob('*.py')):
    name = '.'.join(path.relative_to(ROOT / 'src').with_suffix('').parts)
    names.append(name)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    lines = [name, '=' * len(name), '', f'.. py:module:: {name}', '',
             f'Source: ``{path.relative_to(ROOT).as_posix()}``.', '',
             'Statically extracted signatures and existing docstrings; no module execution.', '',
             prose(tree), ''] + members(tree.body)
    (api / (name + '.rst')).write_text('\n'.join(lines) + '\n', encoding='utf-8')

intro = '''Source API reference
====================

All maintained Python modules under ``src/`` are parsed using Python's AST parser.
Sphinx Napoleon formats existing Google-style docstrings. Signatures, classes,
methods, nested helpers, parameters, returns and exceptions are included where
documented. Missing prose is labelled rather than invented. Imported third-party
members, inherited APIs and dynamically created objects are not expanded; runtime
behaviour and annotation resolution are not verified. Constants are not evaluated.

The organiser copies files on import; COCO preparation and RF-DETR training check
paths on import. Static extraction avoids these side effects for every module.

.. toctree::
   :maxdepth: 1

'''
(api / 'index.rst').write_text(intro + ''.join('   ' + n + '\n' for n in names), encoding='utf-8')


def trim_generated_html(app, exception):
    """Remove template trailing whitespace after a successful HTML build.

    Args:
        app: Sphinx application; outdir identifies generated documentation only.
        exception: Exception or None; failed builds are left available for diagnosis.

    Returns:
        None. Rewrites HTML text without altering elements, links or assets.

    Raises:
        OSError: If generated HTML cannot be read or rewritten.
    """
    if exception is None and app.builder.format == 'html':
        for path in Path(app.outdir).rglob('*.html'):
            text = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n',
                            encoding='utf-8')


def setup(app):
    """Register output formatting without importing or running project modules.

    Args:
        app: Sphinx application receiving the build-finished event handler.
    """
    app.connect('build-finished', trim_generated_html)
