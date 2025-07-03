import json
from pathlib import Path

def split_list_evenly(data:list, n:int) -> list[list]:
    """
    Split `data` into `n` chunks, distributing the remainder items
    so that the first few chunks get one extra element if needed.
    """
    length = len(data)
    base, rem = divmod(length, n)
    chunks = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(data[start:start+size])
        start += size
    return chunks

def main():
    # 1) Load your JSON array:
    # Replace 'input.json' with your filename,
    # or set `raw = [...]` if you already have it in a variable.
    input_path = Path('data.json')
    raw = json.loads(input_path.read_text())

    # 2) Split into 5 chunks:
    parts = split_list_evenly(raw, 5)

    # 3) Write each chunk out:
    for idx, chunk in enumerate(parts, start=1):
        out_path = Path(f'part_{idx}.json')
        out_path.write_text(json.dumps(chunk, indent=2, ensure_ascii=False))
        print(f'Wrote {len(chunk)} items to {out_path}')

if __name__ == '__main__':
    main()
