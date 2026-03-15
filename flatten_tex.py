import re
import os
import sys

def flatten_latex(input_file, output_file):
    input_dir = os.path.dirname(os.path.abspath(input_file))
    
    def process_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Recursive replacement for \input{...}
        # Matches \input{path} or \input{path.tex}
        def replace_input(match):
            included_path = match.group(1)
            if not included_path.endswith('.tex'):
                included_path += '.tex'
            
            # Resolve relative path
            full_path = os.path.join(input_dir, included_path)
            if os.path.exists(full_path):
                print(f"Flattening: {included_path}")
                return f"\n% Begin inclusion: {included_path}\n" + process_file(full_path) + f"\n% End inclusion: {included_path}\n"
            else:
                print(f"Warning: File not found {full_path}")
                return match.group(0) # Keep the original if file not found

        # Regex for \input{filename}
        content = re.sub(r'\\input\{([^}]+)\}', replace_input, content)
        return content

    flattened_content = process_file(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(flattened_content)
    
    print(f"Successfully created flattened file: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flatten_tex.py <input.tex> [output.tex]")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "flattened_" + os.path.basename(in_file)
    
    flatten_latex(in_file, out_file)
