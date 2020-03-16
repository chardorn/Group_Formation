#import sortJupyter as sort

def askForInput(name, size, samples, iterations):
    name_of_excel = input("Input the name of the excel (without .xlsx ending, default exampleList): ")
    if name_of_excel == "":
        name_of_excel = name
    size_of_group = input("Input size of group (default 5): ")
    if size_of_group == "":
        size_of_group = size
    max_samples = input("Number of swaps per iterations (default 500): ")
    if max_samples == "":
        max_samples = samples
    num_iterations = input("Input the number of iterations (default 3): ")
    if num_iterations == "":
        num_iterations = iterations
    print(name_of_excel, size_of_group, max_samples, num_iterations)
    return name_of_excel, size_of_group, max_samples, num_iterations
