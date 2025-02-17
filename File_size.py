import os


def get_files_with_sizes(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                file_list.append((filepath, size))
            except OSError:
                pass
    return file_list


def main():
    try:
        directory = input("Enter the directory path: ")
    except EOFError:
        directory = "../Master_thesis"
    files = get_files_with_sizes(directory)
    files.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 biggest files:")
    for i, (filepath, size) in enumerate(files[:10], start=1):
        print(f"{i}. {filepath} - {size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
