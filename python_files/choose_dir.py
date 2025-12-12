import os


def replace_with_utf8_hex(s: str) -> str:
    forbidden = '*?<>:"|\\'

    result = []
    for ch in s:
        if ch in forbidden:
            # convert to UTF-8 hex (ASCII → one byte)
            hex_value = ch.encode('utf-8').hex().upper()
            result.append(hex_value)
        else:
            result.append(ch)

    return "".join(result)


def list_subdirectories(directory: str):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]


def prompt_for_subdirectory(base_path: str = "data/") -> str:
    """Prompt the user to choose a subdirectory within the base path."""
    subdirs = list_subdirectories(base_path)
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in: {base_path}")

    prompt_lines = ["Choose a folder in data/:"]
    prompt_lines.extend(
        [f"  {idx + 1}. {name}" for idx, name in enumerate(subdirs)])
    prompt_lines.append("Enter number (default 1): ")
    prompt_text = "\n".join(prompt_lines)

    while True:
        choice = input(prompt_text).strip()
        if not choice:
            return os.path.join(base_path, subdirs[0])
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(subdirs):
                return os.path.join(base_path, subdirs[idx])
        print("Invalid selection. Please try again.\n")


def main():
    return prompt_for_subdirectory()


if __name__ == "__main__":
    main()
