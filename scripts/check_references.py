"""Check rendered PDFs for broken cross references.

LaTeX compiles a mistyped control sequence such as ``Section~ef{sec:relevance}``
without complaining, because the surviving text is ordinary prose. The defect is
only visible in the rendered page. This script extracts the text of each built
PDF and fails when a reference artifact reaches the page.
"""
import re
import subprocess
import sys
from pathlib import Path

# Rendered fragments that mean a reference did not resolve or a control
# sequence lost its backslash.
PAGE_PATTERNS = [
    (r'\?\?', 'unresolved reference rendered as ??'),
    (r'(?<![A-Za-z])ef\{', "reference with a stripped backslash (ef{...})"),
    (r'(?<![A-Za-z])(ref|eqref|cite|label|input|includegraphics)\{',
     'control sequence rendered as literal text'),
    (r'\\(ref|cite|label)\b', 'unexpanded macro in the rendered text'),
    (r'(?<![A-Za-z])(Section|Table|Figure|Appendix)~', 'unbreakable space rendered literally'),
]

# Source fragments that produce the defects above. Checked separately so a
# failure names the offending line.
SOURCE_PATTERNS = [
    (r'~\s*ef\{', "'~ef{' should be '~\\ref{'"),
    (r'(?<![\\A-Za-z])(?<!\{)ef\{(sec|tab|fig|eq):', "bare 'ef{' should be '\\ref{'"),
    (r'\r', 'stray carriage return inside a line'),
]


def pdf_text(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        pass
    else:
        with fitz.open(path) as doc:
            return '\n'.join(page.get_text() for page in doc)
    result = subprocess.run(['pdftotext', str(path), '-'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError('no PDF text extractor available (install PyMuPDF '
                           'or pdftotext) for ' + str(path))
    return result.stdout


def check_sources(paths):
    failures = []
    for path in paths:
        raw = path.read_bytes().decode('utf-8', 'replace')
        # Normalize whole-file line endings so only stray in-line CRs are flagged.
        text = raw.replace('\r\n', '\n')
        for number, line in enumerate(text.split('\n'), 1):
            for pattern, message in SOURCE_PATTERNS:
                if re.search(pattern, line):
                    failures.append('{}:{}: {}'.format(path, number, message))
    return failures


def check_pdfs(paths):
    failures = []
    for path in paths:
        text = pdf_text(path)
        for pattern, message in PAGE_PATTERNS:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 60)
                context = ' '.join(text[start:match.end() + 60].split())
                failures.append('{}: {} near "{}"'.format(path, message, context))
    return failures


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / 'research').glob('*.tex'))
    sources += sorted((root / 'research' / 'generated').glob('*.tex'))
    pdfs = [p for p in ((root / 'build/paper/paper.pdf'),
                        (root / 'build/paper/supplement.pdf')) if p.exists()]
    if not pdfs:
        print('No built PDF found under build/paper', file=sys.stderr)
        return 1
    failures = check_sources(sources) + check_pdfs(pdfs)
    if failures:
        for failure in failures:
            print('Reference check failed: ' + failure, file=sys.stderr)
        return 1
    print('Reference check passed for {} sources and {} PDFs'
          .format(len(sources), len(pdfs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
