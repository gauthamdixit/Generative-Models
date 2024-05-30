def decode(message_file):
    with open(message_file, 'r') as file:
        lines = file.readlines()
    
    # Parse the lines into a list of tuples (number, word)
    word_map = []
    for line in lines:
        number, word = line.split()
        word_map.append((int(number), word))
    
    # Sort the list by the number
    word_map.sort()

    # Extract the words at the end of each pyramid line
    decoded_message = []
    step = 1
    index = 0
    while index < len(word_map):
        if index + step - 1 < len(word_map):
            decoded_message.append(word_map[index + step - 1][1])
        else:
            break
        index += step
        step += 1
    
    return ' '.join(decoded_message)

# Example usage:
# Assuming the file 'message.txt' contains the provided input.
decoded_message = decode('message.txt')
print(decoded_message)  # Output: the decoded message