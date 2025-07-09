from bs4 import BeautifulSoup
import os


def extract_sections(html_path):

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove empty or anchor-like spans (already in your code)
    for span in soup.find_all('span'):
        if not span.text.strip() and not span.contents:
            span.decompose()
    # Unwrap span tags that only contain punctuation or colon
    for span in soup.find_all('span'):
        if span.text.strip() in [':', '‣']:
            span.unwrap()
    # Unwrap span tags inside <dd><p> that only contain text (no nested tags)
    for dd in soup.find_all('dd'):
        for p in dd.find_all('p'):
            for span in p.find_all('span'):
                if span.text.strip() and not span.find():
                    span.unwrap()

    for outer_kbd in soup.find_all('kbd'):
        # Check if this <kbd> contains other <kbd> tags inside
        inner_kbds = outer_kbd.find_all('kbd', recursive=False)
        if inner_kbds:
            # Extract text from each inner <kbd>
            combined_text = '-'.join(kbd.get_text(strip=True) for kbd in inner_kbds)
            # Replace the outer <kbd> content with a single combined <kbd>
            new_kbd = soup.new_tag('kbd')
            new_kbd.string = combined_text
            outer_kbd.replace_with(new_kbd)

    # Remove all <a class="headerlink"> tags (the ¶ links)
    for a in soup.find_all('a', class_='headerlink'):
        a.decompose()

    # Remove all <figure> and <figcaption> tags with their content
    for figure in soup.find_all('figure'):
        figure.decompose()
    for figcaption in soup.find_all('figcaption'):
        figcaption.decompose()

    # Replace <img> tags with their alt text or remove if no alt
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '')
        if alt_text:
            img.replace_with(alt_text)
        else:
            img.decompose()

    # Optionally unwrap <div class="refbox admonition"> but keep content
    for div in soup.find_all('div', class_='refbox'):
        div.unwrap()

    for tag in soup.find_all(True):
        if tag.name == 'section':
            attrs = {}
            if 'id' in tag.attrs:
                attrs['id'] = tag.attrs['id']
            tag.attrs = attrs
        elif tag.name in ['div']:
            # Keep class attribute if present
            attrs = {}
            if 'class' in tag.attrs:
                attrs['class'] = tag.attrs['class']
            tag.attrs = attrs
        else:
            tag.attrs = {}

    # Remove empty tags or tags with only whitespace
    for tag in soup.find_all():
        if not tag.text.strip():
            tag.decompose()

    # Extract only top-level <section> tags (not nested inside another section)
    top_level_sections = [sec for sec in soup.find_all('section') if not sec.find_parent('section')]

    cleaned_html = ''.join(str(section) for section in top_level_sections)
    return cleaned_html

def save_cleaned_html(cleaned_html, html_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(html_path))[0]
    output_filename = f"{base_name}_cleaned.html"
    output_path = os.path.join(output_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_html)

def process_all_html_files(base_dir, output_base_dir):
    for root, dirs, files in os.walk(base_dir):
        rel_path = os.path.relpath(root, base_dir)
        output_dir = os.path.join(output_base_dir, rel_path)
        
        for file in files:
            if file.lower().endswith('.html'):
                html_path = os.path.join(root, file)
                print(f"Processing: {html_path}")
                try:
                    cleaned_html = extract_sections(html_path)
                    save_cleaned_html(cleaned_html, html_path, output_dir)
                except Exception as e:
                    print(f"Error processing {html_path}: {e}")

if __name__ == "__main__":
    base_directory = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_project/data/data_types/raw_data/blender_manual_html/blender_manual_v440_en.html"
    output_directory = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_project/data/data_types/processed_data/cleaned_html"
    process_all_html_files(base_directory, output_directory)