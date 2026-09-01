import os
from pathlib import Path


def export_project_for_llm(source_dir, output_dir):
    src = Path(source_dir)
    out = Path(output_dir)

    IGNORE_DIRS = {'.git', '__pycache__', 'venv', '.venv', 'env', '.idea', '.vscode', 'node_modules', 'llm_eksport'}
    IGNORE_FILES = {'.env', '.env.local', 'package-lock.json', 'poetry.lock', 'txtConvertScript.py'}
    IGNORE_EXTS = {'.pyc', '.pyo', '.sqlite3', '.db', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip'}

    out.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file in IGNORE_FILES:
                continue

            file_path = Path(root) / file

            if file_path.suffix.lower() in IGNORE_EXTS:
                continue

            rel_path = file_path.relative_to(src)

            safe_name = str(rel_path).replace(os.sep, '_') + '.txt'
            out_path = out / safe_name

            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    content = f_in.read()

                with open(out_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f"--- ORIGINAL FILE PATH: {rel_path} ---\n\n")
                    f_out.write(content)

                print(f"Skopiowano: {rel_path} -> {safe_name}")
                copied_count += 1

            except UnicodeDecodeError:
                print(f"Pominięto (plik nie-tekstowy): {rel_path}")

    print(f"\nZakończono sukcesem! Wyeksportowano {copied_count} plików do folderu: {out.absolute()}")


if __name__ == "__main__":
    SOURCE_DIRECTORY = "."
    OUTPUT_DIRECTORY = "./llm_eksport"

    export_project_for_llm(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
