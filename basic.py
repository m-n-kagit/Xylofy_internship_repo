# # a = int(input("Enter a number"))
# # b= int(input("Enter a another number"))

# # sum = a+b 
# # print(sum)

# # b = list(map(int,input("Enter numbers separated by space: ").split()))
# # print(b)
# # print(type(b))

# # for i in range(len(b)):
# #     print(b[i])

# # for i in range (1,11):
# #     for j in range(1,i+1):   
# #         print("*", end="")
# #     print()  # Move to the next line after each row 


# # list1 = [ "orange",'benana', "barry","apple"]
# # list1.append("mango")
# # list1.sort()
# # print(list1)

# # Original dictionary
# user_data = {"name": "Alice", "age": 28, "city": "Mumbai", "role": "Engineer"}

# # Keys you want to select
# keep_keys = ["name", "role"]

# # Selected dictionary
# selected_dict = {k: user_data[k] for k in keep_keys if k in user_data}

# print(selected_dict)
# # Output: {'name': 'Alice', 'role': 'Engineer'}



# def my_function(fname, lname):
#   print(fname + " " + lname)

# my_function("Emil")
# def my_function(name):
#   print("Hello", name)

# def my_function(name = " Mohit"): 
#   print("Hello", name)

# a= str(5)
# print(type(a))

# if(a != 5):
#     print("a is not equal to 5")
# else:
#    print("a is equal to 5")

def my_function(a): # pass by reference works for mutable objects like lists,
    # but not for immutable objects like integers.
  a= a+5 

f=12
my_function(f)
print(f)  # Output: 12, because integers are immutable and the original value of f is not changed


# def swap(x,y):
#     temp = x
#     x = y
#     y = temp

# x = 10
# y=5
# swap(x,y)
# print("x:", x)  # Output: 10
# print("y:", y)  # Output: 5, because integers are immutable and the original values of x and y are not changed

# x=5
# y=6
# def swap(x,y):
#     return y,x
# y,x = swap(x,y)
# print("x:", x)  # Output: 6
# print("y:", y)  # Output: 5
