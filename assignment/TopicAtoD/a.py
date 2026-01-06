# 6. Write a Python program to check the type of variables (int, float, string, boolean).
def check_variable_type(var):
    if isinstance(var, int):
        return "The variable is of type: Integer"
    elif isinstance(var, float):
        return "The variable is of type: Float"
    elif isinstance(var, str):
        return "The variable is of type: String"
    elif isinstance(var, bool):
        return "The variable is of type: Boolean"
    else:
        return "The variable is of an unknown type"
    
# Example usage
variables = [42, 3.14, "Hello, World!", True, None]
for var in variables:
    print(check_variable_type(var))

user_input = input("Enter a value: ")
print(check_variable_type(user_input))

#7. Take input from the user for name and age. Print a greeting message using f-string formatting.
name = input("Enter your name: ")
age = input("Enter your age: ") 
greeting_message = f"Hello, {name}! You are {age} years old."
print(greeting_message)


#8. Differentiate between single-line, multi-line comments, and docstrings in Python with examples.

# Single-line comment: This is a comment that occupies a single line.
"""
Multi-line comment:
This is a comment that spans multiple lines.
It is often used for longer explanations or documentation.
""" 
def example_function():
    """ 
    Docstring: This is a special type of comment used to describe a function, class, or module.
    It is enclosed in triple quotes and can span multiple lines.
    """
    pass

#9. Write a program to calculate the area of a circle. (Hint: use math.pi)
import math
def calculate_circle_area(radius):
    area = math.pi * (radius ** 2)
    return area

# Example usage
radius = float(input("Enter the radius of the circle: "))
area = calculate_circle_area(radius)
print(f"The area of the circle with radius {radius} is: {area}")


#10. Explain type casting in Python with examples (implicit vs explicit).

# Type casting in Python refers to the conversion of one data type to another.
# Implicit type casting: Python automatically converts one data type to another without user intervention.
x = 5          # Integer
y = 2.0        # Float
result = x + y  # Implicitly converts x to float
print(f"Implicit type casting result: {result} (type: {type(result)})")

# Explicit type casting: The user manually converts one data type to another using built-in functions.
a = "10"       # String
b = int(a)     # Explicitly converts string to integer
print(f"Explicit type casting result: {b} (type: {type(b)})")   

# 11. Write a program to check if a given number is positive, negative, or zero using if-elif-else.
number = float(input("Enter a number: "))
if number > 0:
    print("The number is positive.")    
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")

#12. Write a program to print the first 10 natural numbers using a for loop.
print("The first 10 natural numbers are:")
for i in range(1, 11):
    print(i)

# 13. Write a program that prints the multiplication table of a number using a while loop.
num = int(input("Enter a number to print its multiplication table: "))
print(f"Multiplication table of {num}:")
i = 1
while i <= 10:
    print(f"{num} x {i} = {num * i}")
    i += 1


# 14 Aarmstrong number is a number that is equal to the sum of its own digits each raised to the power of the number of digits.
num = input("enter Number")
res=0
for i in num:
    # print(i)
    res += int(i) ** len(num)
    print(res)
    
if(res == int(num)):
    print("Am")
else:
    print("Not Am")

#15. Write a program to find the largest among three numbers using nested if-else.
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))   
c = float(input("Enter third number: "))
if a >= b:
    if a >= c:
        largest = a
    else:
        largest = c