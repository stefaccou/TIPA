# Test file to print hello world and save it to a text file
print("Hello World")

# Save the output to a text file
with open("output.txt", "w") as file:
    file.write("Hello World\n")
