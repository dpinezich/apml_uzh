#!/usr/bin/env python3
"""
Convert chapter markdown files to Slidev format.
Handles two input formats:
  - reveal-md: has title in frontmatter, <!-- .slide: class="slide-title/end" -->, Note: blocks, class="fragment"
  - Marp:      has marp: true frontmatter, no special markers, no Notes/fragments
"""
import re
import sys
from pathlib import Path

SLIDE_FILES = [
    "1-introduction/01-slides/ch01_slides.md",
    "2-selection_cleaning_preparing/01-slides/ch02_slides.md",
    "3-supervised_learning/01-slides/ch03_slides.md",
    "3-supervised_learning/01-slides/ch04_slides.md",
    "3-supervised_learning/01-slides/ch05_slides.md",
    "3-supervised_learning/01-slides/ch06_slides.md",
    "4-unsupervised_learning/01-slides/ch07_slides.md",
    "4-unsupervised_learning/01-slides/ch08_slides.md",
    "4-unsupervised_learning/01-slides/ch09_slides.md",
    "5-reinforcement_learning/01-slides/ch10_slides.md",
    "5-reinforcement_learning/01-slides/ch11_slides.md",
    "6-capstone_ml/01-slides/ch12_slides.md",
]


def convert_fragments(text: str) -> str:
    """Convert class="fragment" to v-click on HTML elements."""
    text = re.sub(r'(<\w+)\s+class="fragment"', r'\1 v-click', text)
    return text


def convert_notes(text: str) -> str:
    """Convert Note: blocks to HTML comments."""
    lines = text.split('\n')
    result = []
    in_note = False
    note_lines = []

    for line in lines:
        if re.match(r'^Note:\s*$', line):
            in_note = True
            note_lines = []
        elif in_note:
            note_lines.append(line)
        else:
            result.append(line)

    if in_note and note_lines:
        while note_lines and not note_lines[-1].strip():
            note_lines.pop()
        result.append('')
        result.append('<!--')
        result.extend(note_lines)
        result.append('-->')

    return '\n'.join(result)


def build_output(title: str, cover_content: str, regular_slides: list, end_content: str) -> str:
    """Assemble the final Slidev output."""
    # Escape double quotes in title for YAML
    title_yaml = title.replace('"', '\\"')
    global_front = f'---\nlayout: cover\ntitle: "{title_yaml}"\n---'

    sections = []

    # Slide 1: cover
    if cover_content:
        sections.append(global_front + '\n\n' + cover_content.strip())
    else:
        sections.append(global_front)

    # Middle slides
    for slide in regular_slides:
        sections.append(slide.strip())

    # End slide
    if end_content:
        # Special join: end slide has its own frontmatter
        result = '\n\n---\n\n'.join(sections)
        result += '\n\n---\nlayout: end\n---\n\n' + end_content.strip() + '\n'
    else:
        result = '\n\n---\n\n'.join(sections) + '\n'

    return result


def convert_reveal_md(src: str) -> str:
    """Convert reveal-md format (title frontmatter + slide-title/end markers)."""
    raw_slides = re.split(r'\n---\n', src)

    frontmatter_block = raw_slides[0]
    title_match = re.search(r'^title:\s*(.+?)\s*$', frontmatter_block, re.MULTILINE)
    title = title_match.group(1).strip('"\'') if title_match else 'Slides'

    cover_content = ''
    regular_slides = []
    end_content = ''

    for slide in raw_slides[1:]:
        slide = slide.strip('\n')

        is_title = bool(re.search(r'<!--\s*\.slide:\s*class="slide-title"\s*-->', slide))
        is_end = bool(re.search(r'<!--\s*\.slide:\s*class="slide-end"\s*-->', slide))

        if is_title:
            slide = re.sub(r'<!--\s*\.slide:\s*class="slide-title"\s*-->\s*\n?', '', slide).strip()
            slide = convert_notes(slide)
            slide = convert_fragments(slide)
            cover_content = slide

        elif is_end:
            slide = re.sub(r'<!--\s*\.slide:\s*class="slide-end"\s*-->\s*\n?', '', slide).strip()
            slide = convert_notes(slide)
            slide = convert_fragments(slide)
            end_content = slide

        else:
            slide = convert_notes(slide)
            slide = convert_fragments(slide)
            regular_slides.append(slide)

    return build_output(title, cover_content, regular_slides, end_content)


def convert_marp(src: str, path: Path = None) -> str:
    """Convert Marp format (marp: true frontmatter, plain slides)."""
    raw_slides = re.split(r'\n---\n', src)

    # raw_slides[0] is the Marp frontmatter block
    slides = [s.strip('\n') for s in raw_slides[1:] if s.strip()]

    if not slides:
        return src

    # Determine chapter number from filename (ch03_slides.md → 03)
    ch_prefix = ''
    if path:
        m = re.search(r'ch(\d+)_slides', path.name)
        if m:
            ch_num = int(m.group(1))
            ch_prefix = f'Ch{ch_num:02d} — '

    # Title: first h1 in first slide (with ChXX prefix)
    title = 'Slides'
    first_slide = slides[0]
    h1_match = re.search(r'^# (.+)$', first_slide, re.MULTILINE)
    if h1_match:
        h1 = h1_match.group(1).strip()
        # For ch12, combine h1 + h2 if h1 is just "Chapter 12"
        if re.match(r'^Chapter \d+$', h1):
            h2_match = re.search(r'^## (.+)$', first_slide, re.MULTILINE)
            h1 = h2_match.group(1).strip() if h2_match else h1
        title = ch_prefix + h1

    cover_content = first_slide

    # Detect end slide: last slide matches "# Next: Chapter" or capstone pattern
    end_content = ''
    regular_slides = slides[1:]

    if regular_slides:
        last = regular_slides[-1]
        # End slide patterns: "# Next:" or the capstone action slide
        if re.search(r'^# Next:', last, re.MULTILINE) or re.search(r'^# Now —', last, re.MULTILINE):
            end_content = last
            regular_slides = regular_slides[:-1]

    return build_output(title, cover_content, regular_slides, end_content)


def convert_file(path: Path) -> None:
    src = path.read_text(encoding='utf-8')

    # Detect format
    if re.search(r'^marp:\s*true', src, re.MULTILINE):
        result = convert_marp(src, path)
        fmt = 'Marp'
    else:
        result = convert_reveal_md(src)
        fmt = 'reveal-md'

    path.write_text(result, encoding='utf-8')
    print(f'  converted [{fmt}]: {path.name}')


def main():
    root = Path(__file__).parent
    dry_run = '--dry-run' in sys.argv

    for rel_path in SLIDE_FILES:
        p = root / rel_path
        if not p.exists():
            print(f'  MISSING:   {p}')
            continue
        if dry_run:
            src = p.read_text()
            fmt = 'Marp' if re.search(r'^marp:\s*true', src, re.MULTILINE) else 'reveal-md'
            print(f'  would convert [{fmt}]: {p.name}')
        else:
            convert_file(p)

    print('Done.')


if __name__ == '__main__':
    main()
