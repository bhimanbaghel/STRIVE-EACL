#!/usr/bin/env python3
"""
Convert JSON prompt files to LaTeX figure* environments with tcolorbox + Verbatim.
Automatically splits prompts across multiple figures if they exceed a line limit.

Usage:
  python prompt_to_latex.py <prompt_json> [--lines-per-page 50] [--output-dir latex_figures]
  python prompt_to_latex.py prompts/v7_agent.json --lines-per-page 45 --color blue
  python prompt_to_latex.py --all --prompt-dir prompts/ --output-dir latex_figures/ --color slate

Color presets (--color): neutral, blue, slate, teal, cream

Required LaTeX packages (add to your preamble):
  \\usepackage{tcolorbox}
  \\usepackage{fancyvrb}
"""

import json
import os
import sys
import argparse
import textwrap
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt display names and LaTeX label prefixes
# ---------------------------------------------------------------------------
PROMPT_NAMES = {
    "v4": ("Single-Shot Base Generation Prompt", "ss-base"),
    "v5": ("Single-Shot Visual Generation Prompt", "ss-visual"),
    "evaluator_v2": ("Evaluator Prompt", "evaluator"),
    "refinement_v1": ("Iterative Refinement Prompt", "refinement"),
}

# tcolorbox color presets tuned for appendix / prompt figures in academic papers.
# All use light backgrounds and moderate frames for print and grayscale safety.
COLOR_PRESETS = {
    "neutral": {
        "colback": "gray!5",
        "colframe": "gray!75",
        "description": "Default gray; safe for any venue (original style)",
    },
    "blue": {
        "colback": "blue!4",
        "colframe": "blue!55!black",
        "description": "Classic academic blue accent; common in CS/ML papers",
    },
    "slate": {
        "colback": "black!3",
        "colframe": "black!50",
        "description": "Minimal blue-gray; best for B&W printing and strict venues",
    },
    "teal": {
        "colback": "teal!4",
        "colframe": "teal!60!black",
        "description": "Modern teal accent; readable on screen and in color print",
    },
    "cream": {
        "colback": "yellow!6!white",
        "colframe": "orange!45!black",
        "description": "Warm parchment tone; easy on the eyes for long appendix blocks",
    },
}

DEFAULT_COLOR_PRESET = "neutral"


def get_color_preset(name):
    """Return tcolorbox color options for a named preset."""
    if name not in COLOR_PRESETS:
        valid = ", ".join(sorted(COLOR_PRESETS))
        raise ValueError(f"Unknown color preset '{name}'. Choose from: {valid}")
    return COLOR_PRESETS[name]


def format_tcolorbox_colors(preset):
    """Format colback/colframe options for a tcolorbox."""
    return f"colback={preset['colback']}, colframe={preset['colframe']}"


def get_display_info(filename):
    """Get display name and label prefix from filename."""
    stem = Path(filename).stem
    if stem in PROMPT_NAMES:
        return PROMPT_NAMES[stem]
    # Fallback: clean up the filename
    display = stem.replace("_", " ").title() + " Prompt"
    label = stem.replace("_", "-")
    return display, label


def load_prompt_lines(json_path, max_line_width=90):
    """Load a prompt JSON and return formatted lines for Verbatim display."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_lines = []

    # Add system prompt if present
    if 'system' in data:
        all_lines.append("=== SYSTEM PROMPT ===")
        all_lines.append("")
        for line in data['system']:
            if not line.strip():
                all_lines.append("")
            elif len(line) > max_line_width:
                wrapped = textwrap.fill(line, width=max_line_width,
                                        break_long_words=False,
                                        break_on_hyphens=False)
                all_lines.extend(wrapped.split('\n'))
            else:
                all_lines.append(line)
        all_lines.append("")

    # Add user template if present
    if 'userTemplate' in data:
        all_lines.append("=== USER TEMPLATE ===")
        all_lines.append("")
        for line in data['userTemplate']:
            if not line.strip():
                all_lines.append("")
                continue

            # Wrap long lines while preserving indentation
            indent = len(line) - len(line.lstrip())
            if len(line) > max_line_width:
                wrapped = textwrap.fill(line.strip(),
                                        width=max_line_width - indent,
                                        initial_indent=' ' * indent,
                                        subsequent_indent=' ' * (indent + 2),
                                        break_long_words=False,
                                        break_on_hyphens=False)
                all_lines.extend(wrapped.split('\n'))
            else:
                all_lines.append(line)

    # Fallback: try 'prompt' key
    if 'prompt' in data and not all_lines:
        if isinstance(data['prompt'], list):
            for line in data['prompt']:
                if not line.strip():
                    all_lines.append("")
                elif len(line) > max_line_width:
                    wrapped = textwrap.fill(line, width=max_line_width,
                                            break_long_words=False,
                                            break_on_hyphens=False)
                    all_lines.extend(wrapped.split('\n'))
                else:
                    all_lines.append(line)
        else:
            wrapped = textwrap.fill(data['prompt'], width=max_line_width,
                                    break_long_words=False,
                                    break_on_hyphens=False)
            all_lines.extend(wrapped.split('\n'))

    return all_lines


def escape_latex_verbatim(line):
    """Escape characters that cause issues inside Verbatim environments."""
    # Verbatim handles most things, but we need to be careful with
    # unbalanced braces in some edge cases. For fancyvrb Verbatim,
    # content is taken literally, so minimal escaping is needed.
    return line


def split_into_chunks(lines, lines_per_page):
    """Split lines into chunks of at most lines_per_page lines.
    Tries to split at section boundaries (--- or === lines) when possible."""
    if len(lines) <= lines_per_page:
        return [lines]

    chunks = []
    current_chunk = []

    for i, line in enumerate(lines):
        current_chunk.append(line)

        if len(current_chunk) >= lines_per_page:
            # Try to find a nearby section break to split at
            best_split = len(current_chunk)
            # Look back up to 10 lines for a section break
            for j in range(len(current_chunk) - 1, max(len(current_chunk) - 11, 0), -1):
                candidate = current_chunk[j].strip()
                if candidate.startswith('---') or candidate.startswith('===') or candidate == '':
                    best_split = j
                    break

            if best_split < len(current_chunk) and best_split > lines_per_page - 15:
                # Split at the section break
                chunks.append(current_chunk[:best_split])
                current_chunk = current_chunk[best_split:]
            else:
                # No good break point, split at the limit
                chunks.append(current_chunk[:])
                current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_latex_figure(lines, display_name, label_prefix, part_num=None,
                          total_parts=None, is_continuation=False,
                          color_preset=None):
    """Generate a LaTeX figure* environment string."""

    # Build title
    if total_parts and total_parts > 1:
        title_text = f"{display_name} (Part {part_num}/{total_parts})"
    else:
        title_text = display_name

    # Build label
    if total_parts and total_parts > 1:
        label = f"fig:{label_prefix}-part{part_num}"
    else:
        label = f"fig:{label_prefix}"

    # Build caption
    if is_continuation:
        caption = f"{display_name} (continued from previous page)."
    elif total_parts and total_parts > 1:
        caption = f"{display_name}. Continued in Figure~\\ref{{fig:{label_prefix}-part{part_num + 1}}}."
    else:
        caption = f"{display_name}."

    # Build the Verbatim content
    verbatim_content = '\n'.join(escape_latex_verbatim(l) for l in lines)

    if color_preset is None:
        color_preset = COLOR_PRESETS[DEFAULT_COLOR_PRESET]
    tcolorbox_colors = format_tcolorbox_colors(color_preset)

    # Assemble LaTeX
    latex = f"""\\begin{{figure*}}[t]
\\centering
\\begin{{tcolorbox}}[width=\\textwidth, {tcolorbox_colors},
                  title={{{title_text}}}]
\\begin{{Verbatim}}[fontsize=\\footnotesize]
{verbatim_content}
\\end{{Verbatim}}
\\end{{tcolorbox}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure*}}
"""
    return latex


def process_prompt(json_path, lines_per_page=50, max_line_width=90, output_dir="latex_figures",
                   color_preset=None):
    """Process a single prompt JSON file and generate LaTeX figure files."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_name, label_prefix = get_display_info(json_path.name)
    lines = load_prompt_lines(json_path, max_line_width)

    preset_name = next(
        (k for k, v in COLOR_PRESETS.items() if v is color_preset),
        DEFAULT_COLOR_PRESET,
    )

    print(f"\nProcessing: {json_path.name}")
    print(f"  Display name: {display_name}")
    print(f"  Color preset: {preset_name}")
    print(f"  Total lines: {len(lines)}")

    chunks = split_into_chunks(lines, lines_per_page)
    total_parts = len(chunks)

    print(f"  Parts: {total_parts} (lines per page: {lines_per_page})")

    generated_files = []

    for i, chunk in enumerate(chunks):
        part_num = i + 1
        is_continuation = i > 0
        is_last = i == total_parts - 1

        latex = generate_latex_figure(
            lines=chunk,
            display_name=display_name,
            label_prefix=label_prefix,
            part_num=part_num if total_parts > 1 else None,
            total_parts=total_parts if total_parts > 1 else None,
            is_continuation=is_continuation,
            color_preset=color_preset,
        )

        # Generate filename
        if total_parts > 1:
            filename = f"{label_prefix}_part{part_num}.tex"
        else:
            filename = f"{label_prefix}.tex"

        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex)

        generated_files.append(filepath)
        print(f"  Generated: {filepath} ({len(chunk)} lines)")

    # Generate an include file that inputs all parts in order
    if total_parts > 1:
        include_path = output_dir / f"{label_prefix}_all.tex"
        with open(include_path, 'w') as f:
            f.write(f"% Auto-generated: include all parts of {display_name}\n")
            for gf in generated_files:
                f.write(f"\\input{{{gf.as_posix()}}}\n")
        print(f"  Include file: {include_path}")
        generated_files.append(include_path)

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON prompt files to LaTeX figure* environments"
    )

    # Input options (required unless --list-colors)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("prompt", nargs="?", default=None,
                             help="Path to a single prompt JSON file")
    input_group.add_argument("--all", action="store_true",
                             help="Process all JSON files in the prompt directory")

    # Configuration
    parser.add_argument("--lines-per-page", type=int, default=70,
                        help="Maximum lines per figure (default: 70)")
    parser.add_argument("--max-line-width", type=int, default=98,
                        help="Maximum characters per line before wrapping (default: 98)")
    parser.add_argument("--output-dir", default="latex_figures",
                        help="Output directory for .tex files (default: latex_figures)")
    parser.add_argument("--prompt-dir", default="prompts",
                        help="Directory containing prompt JSON files (for --all)")
    color_help = (
        "tcolorbox color preset (default: %(default)s). "
        + "; ".join(
            f"{name}: {preset['description']}"
            for name, preset in sorted(COLOR_PRESETS.items())
        )
    )
    parser.add_argument(
        "--color",
        choices=sorted(COLOR_PRESETS),
        default=DEFAULT_COLOR_PRESET,
        metavar="PRESET",
        help=color_help,
    )
    parser.add_argument(
        "--list-colors",
        action="store_true",
        help="List available color presets and exit",
    )

    args = parser.parse_args()

    if args.list_colors:
        print("Available color presets:\n")
        for name in sorted(COLOR_PRESETS):
            preset = COLOR_PRESETS[name]
            default = " (default)" if name == DEFAULT_COLOR_PRESET else ""
            print(f"  {name}{default}")
            print(f"    {preset['description']}")
            print(f"    colback={preset['colback']}, colframe={preset['colframe']}\n")
        sys.exit(0)

    if not args.all and not args.prompt:
        parser.error("Provide a prompt file path, use --all, or use --list-colors")

    color_preset = get_color_preset(args.color)

    if args.all:
        prompt_dir = Path(args.prompt_dir)
        if not prompt_dir.exists():
            print(f"Error: Prompt directory not found: {prompt_dir}")
            sys.exit(1)

        json_files = sorted(prompt_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {prompt_dir}")
            sys.exit(1)

        print(f"Found {len(json_files)} prompt files in {prompt_dir}")
        all_files = []
        for jf in json_files:
            files = process_prompt(
                jf, args.lines_per_page, args.max_line_width, args.output_dir, color_preset
            )
            all_files.extend(files)

        # Generate master include file
        master_path = Path(args.output_dir) / "all_prompts.tex"
        with open(master_path, 'w') as f:
            f.write("% Auto-generated: include all prompt figures\n")
            f.write("% Add to your appendix: \\input{latex_figures/all_prompts}\n\n")
            for jf in json_files:
                _, label_prefix = get_display_info(jf.name)
                include_file = Path(args.output_dir) / f"{label_prefix}_all.tex"
                single_file = Path(args.output_dir) / f"{label_prefix}.tex"
                if include_file.exists():
                    f.write(f"\\input{{{include_file.as_posix()}}}\n")
                elif single_file.exists():
                    f.write(f"\\input{{{single_file.as_posix()}}}\n")

        print(f"\nMaster include file: {master_path}")
        print(f"Total files generated: {len(all_files)}")

    else:
        if not args.prompt:
            print("Error: Provide a prompt file path or use --all")
            sys.exit(1)

        prompt_path = Path(args.prompt)
        if not prompt_path.exists():
            print(f"Error: File not found: {prompt_path}")
            sys.exit(1)

        process_prompt(
            prompt_path, args.lines_per_page, args.max_line_width, args.output_dir, color_preset
        )


if __name__ == "__main__":
    main()