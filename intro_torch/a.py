class MyClass:
    def __main__ (self,b):
        print(b)

    def __init__(self, a):
        self.a = a

    def print_a(self):
        print(self.a)

# Creating an instance of the class and passing the variable 'a'
a = 3
my_object = MyClass(a)

# Calling a method of the object to print the value of 'a'
my_object.print_a()

my_object(a)
