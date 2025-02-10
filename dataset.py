import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

def process_examples(base_dir):
    index_path = os.path.join(base_dir, 'indexes.examples.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        index_soup = BeautifulSoup(f, 'html.parser')

    data = []
    links = index_soup.find_all('a', class_='index')
    
    for link in tqdm(links, desc="Processing examples", unit="example"):
        href = link.get('href')
        if not href:
            continue

        parts = href.split('#', 1)
        file_path = os.path.join(base_dir, parts[0])
        fragment = parts[1] if len(parts) > 1 else None

        if not os.path.exists(file_path):
            tqdm.write(f"⚠️ File not found: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            file_soup = BeautifulSoup(f, 'html.parser')

        example_div = None
        if fragment:
            example_div = file_soup.find('div', class_='example', id=fragment)
        else:
            example_div = file_soup.find('div', class_='example')

        if not example_div:
            tqdm.write(f"⛔ Example div not found in {file_path} (fragment: {fragment})")
            continue

        # Extract input description
        input_p = example_div.find('p')
        input_text = input_p.get_text().strip() if input_p else ''

        # Extract code content with proper whitespace handling
        output_code = ''
        example_contents = example_div.find('div', class_='example-contents')
        if example_contents:
            phpcode_div = example_contents.find('div', class_='phpcode')
            if phpcode_div:
                code_tag = phpcode_div.find('code')
                if code_tag:
                    # Preserve original whitespace formatting
                    output_code = '\n'.join(
                        [line.rstrip() for line in code_tag.get_text('\n').splitlines()]
                    ).strip()

        if input_text and output_code:
            data.append({
                'input': input_text,
                'output': output_code
            })

    return data

def main():
    base_directory = 'php-chunked-xhtml'
    training_data = process_examples(base_directory)

    with open('training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully processed {len(training_data)} examples")
    print(f"💾 Saved to training_data.json")

if __name__ == '__main__':
    main()
