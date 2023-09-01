import os

def extract_sentence_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sentence = content.split('.')[0].strip()  # Till the first full stop.
    sentence = sentence.split('#')[0].strip()  # Till the first full stop.
    return sentence

def extract_text_prompts(folder_path, output_folder_path, output_file_path):
    text_prompts = []

    os.makedirs(output_folder_path, exist_ok=True)

    files = os.listdir(folder_path)
    numeric_files = sorted([file for file in files if file.startswith('0') and file.endswith('.txt')], key=lambda x: int(x.split('.')[0]))
    m_files = sorted([file for file in files if file.startswith('M') and file.endswith('.txt')], key=lambda x: int(x.split('.')[0][1:]))

    with open(output_file_path, 'w') as output_file:
        for file_name in numeric_files + m_files:
            file_path = os.path.join(folder_path, file_name)
            sentence = extract_sentence_from_file(file_path)
            text_prompts.append(sentence)

            # Create individual text files
            output_file_individual = os.path.join(output_folder_path, f"{os.path.splitext(file_name)[0]}.txt")
            with open(output_file_individual, 'w') as output_individual:
                output_individual.write(sentence)

        # Write to the consolidated text file after collecting all sentences
        output_file.write('\n'.join(text_prompts))

    print(f"Text prompts extracted and saved to '{output_folder_path}' and consolidated in '{output_file_path}'.")

# Usage example
folder_path = '/data/tejasr20/motion-diffusion-model/dataset/HumanML3D/texts'
output_folder = '/data/tejasr20/motion-diffusion-model/dataset/HumanML3D/smpl_texts'
output_file = '/data/tejasr20/motion-diffusion-model/dataset/HumanML3D/full_text.txt'

extract_text_prompts(folder_path, output_folder, output_file)
