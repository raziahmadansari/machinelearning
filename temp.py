def sum(n1, n2):
    n1_int = convert_integer(n1)
    n2_int = convert_integer(n2)

    result = n1_int + n2_int

    return result

def convert_integer(n_string):
    converted_integer = int(n_string)
    return converted_integer

answer = sum("1", "2")
