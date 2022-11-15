### Write a function that can calculate the area of a circle ###


# Import the math module
import math
# The formula for the area of a circle is: pi * r^2
# Define a function that takes the radius as an argument
def area_of_circle(radius):
    # Calculate the area of the circle
    area = math.pi * radius ** 2
    # Return the area
    return area
# Write the main function
def main():
    # Call the area_of_circle function and print the result
    print(area_of_circle(5))
# Call the main function
# This is the entry point of the program
# The main function will be called when the program is executed
# The main function will not be called when the program is imported
# This is because the __name__ variable will be set to the name of the module
# when the program is imported
if __name__ == "__main__":
    main()   
