#import sortJupyter as sort

def askForInput():
    name_of_excel= input("Input the name of the excel (without .xlsx ending): ")
    size_of_group = input("Input size of group: ")
    max_samples = input("Number of swaps per iterations (default: 500): ")
    num_iterations = input("Input the number of iterations (default: 3): ")
    return name_of_excel, size_of_group, max_samples, num_iterations
