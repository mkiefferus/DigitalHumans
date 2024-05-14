import os
import ast

def extract_functions_and_docstrings(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            docstring = ast.get_docstring(node)
            functions.append((func_name, docstring))
    
    return functions

def generate_function_descriptions(directory, extensions=[".py"]):
    descriptions = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                functions = extract_functions_and_docstrings(file_path)
                if functions:
                    descriptions.append(f"\n### {file}\n")
                    for func_name, docstring in functions:
                        descriptions.append(f"**{func_name}**\n")
                        if docstring:
                            descriptions.append(f"```\n{docstring}\n```\n")
                        else:
                            descriptions.append("No docstring provided.\n")
    
    return "\n".join(descriptions)

def append_to_readme(readme_path, descriptions):
    with open(readme_path, 'a') as readme_file:
        readme_file.write("\n## Function Descriptions\n")
        readme_file.write(descriptions)

def main():
    readme_path = "TEST_README.md"
    project_directory = "prompt_enhancement_models/"
    
    descriptions = generate_function_descriptions(project_directory)
    append_to_readme(readme_path, descriptions)

if __name__ == "__main__":
    main()
