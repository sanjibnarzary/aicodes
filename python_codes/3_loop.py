# Write a program that prints the numbers from 1 to 10
# using for loop
# 

# Write the main function
def main():
    # Use a for loop to iterate from 1 to 10
    for i in range(1, 11):
        # Print the value of i
        print(i)
    # use a while loop to iterate from 1 to 10
    i = 1
    while i <= 10:
        # Print the value of i
        print(i)
        # Increment i
        i += 1
    # use do while loop to iterate from 1 to 10
    i = 1
    while True:
        # Print the value of i
        print(i)
        #print a string is inside the loop
        print("i = " + str(i))
      
        # Increment i
        i += 1
        # Check if i is greater than 10
        if i > 10:
            # Break out of the loop
            break
    # use lambda function to iterate from 1 to 10
    list(map(lambda x: print(x), range(1, 11)))
    # use list comprehension to iterate from 1 to 10
    [print(x) for x in range(1, 11)]
    # use recursion to iterate from 1 to 10
    def print_numbers(n):
        # print the value of n with string function
        print(str(n))
        # Check if n is greater than 10
        if n > 10:
            # Return from the function
            return
        # Print the value of n
        print(n)
        # Call the function again with n + 1
        print_numbers(n + 1)
    # Call the function
    print_numbers(1)

# Call the main function
# This is the entry point of the program
# The main function will be called when the program is executed
# The main function will not be called when the program is imported
# This is because the __name__ variable will be set to the name of the module
# when the program is imported
if __name__ == "__main__":
    main()