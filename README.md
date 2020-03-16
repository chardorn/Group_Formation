# Group Formation
This program allows you to sort individuals into intentionally diverse groups based on certain traits. 

As input, it takes an excel spreadsheet preformatted as specified below.
As output, it produces another excel, titled Sorted_Output with groups. 
The parameters (and defaults) are the name of the excel (exampleList.xlsx), group size (5), max # of swaps per iteration (500), and the number of iterations (3). 
    
# How to Use

Download or clone this repository onto your computer. 
Add your excel file to the Group_Formation folder, and ensure that it is formatted as detaliled below. The name of the excel should not contain any spaces to avoid error. 
After installing the neccesary dependencies (listed below), the file can be run by navigating to the folder and running:
```
python3 sort.py
```
Then the following should pop up (one at a time) and you should input a numerical value and hit enter.

```
Input the name of the excel (without .xlsx ending): 
Input size of group: 
Number of swaps per iterations (default: 500): 
Input the number of iterations (default: 3): 
```

# System Prerequisites:
- [Python3](https://www.python.org/downloads/)
- Packages:
    - numpy
    - pandas
    - openpyxl
    - matplotlib.pyplot
    

# Format Excel Data

Example:

| Number | Gender | Ethnicity | Residency | Majors                      | Availability | Roommates      |
|--------|--------|-----------|-----------|-----------------------------|--------------|----------------|
| 123    | F      | Hispanic  | IS        | Computer Science, Economics | T, T, F      | 126, 131       |
| 124    | M      | White     | OOS       | English, PWAD               | T, T, T      |                |
| ...    | ...    | ...       | ...       | ...                         | ...          | ...            |

The top row of the table, of course, should be the headers. The Name/Number must be the first column followed by any number of headers. 

### Headers
Here are the possible headers and their respective functions:
- **"Name" or "Number"**: this is simply the name or identifying number of each person
- **"Gender" or "Sex"**: this is a binary sort which aiming to get as close to equal amounts of each gender in every group
    TODO: Allow for more input of more than two genders. For now, for optimization, group in any nonbinary identifiers with the smaller of the two groups. For example, if the group was predominantly female, the labels could be "Female" and "Nonfemale"

*The following five functions aim to create as diverse as a group of people, with as many different traits in each group. For example if there are five regions, it will optimize for the highest number of different regions in each group. The most ideal group would have all five members from each of the five different regions. The same applies for ethnicity and residency and majors.*
- **"Ethnicity" or "Race"**
- **"Residency"**: intended use is for In-State vs Out-of-State vs International
- **"Region"**: intended use is for different counties within the state
- **"Rural" or "Urban**
- **"Major" or "Majors"**
         *in order to list more than one major, list them in the one cell, with a comma in between (up to 2)

*The following three functions are hard checks. A group cannot be formed if it violates one of the following checks.*

- **"School" or "High School"**: the program will not allow two students from the same school to be in the same group
- **"Availability" or "Available"**: this allows to check to see if all members of a group have at least one time when they are all available. The data should be in the form of a list of "T" or "F". For example, if a indivdual is avaialable for only the first three time slots, their corresponding cell would be "T, T, T, F, F"
- **"Roommates" or "Incompatable"**: this ensures that no two incompatible members are in the same group. The data should be in the form of names or numbers seperated be commas. For example, if an indivudal was incomptable with two people with the numbers 123 and 124, they're cell would be "123, 124". 

### Notes:
It is very important to make sure that the input data is formatted correctly. Take care to check the following to avoid possible issues:
- No extranesous space before or after a word
- Consistent capitalization or abbreviations
- Name/Number must exactly match when used in Incompatables

The weights of each variable can be adjusted within the document if desired. 
