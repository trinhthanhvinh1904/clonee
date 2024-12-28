def process_string(input_str):
    input_str = input_str.replace(" ", "")
    
    input_str = input_str.replace(",", ".")
    
    if input_str.count(".") > 1:
        parts = input_str.split(".")
        input_str = "".join(parts[:-1]) + "." + parts[-1]
    
    similar_characters = {'g': '9','o':'0','O':'0', 'a': '0', 'j': '1', 'l': '1', '(': '1', ')': '1', 'b': '6', 'p': '8', 'f': '8', 'z': '2', '2': '2'}
    input_str = "".join(similar_characters.get(char, char) for char in input_str)
    
    try:
        number = float(input_str)
        while number > 20:
            number /= 10
        return number
    except ValueError:
        return input_str

test = "og"
print(f"Input: {test} -> Output: {process_string(test)}")
