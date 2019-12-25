# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:53:30 2019

@author: GS65 8RF
"""
 #__init__,__call__,*args,**kwargs
 #for __call__,__call__ makes the instance of a class callable. Why would it be required?
 #Technically __init__ is called once by __new__ when object is created, so that it can be initialized.
 #But there are many scenarios where you might want to redefine your object, say you are done with your object,
 # and may find a need for a new object.With __call__ you can redefine the same object as if it were new.
 #This is just one case, there can be many more.
 #for ** you must declare key=value to automatically know **.If declare key:value, it thought as *.For this situation
 #say as ** when you call.I demostrated it as follow
class Employee:
    id = 0
    name = ""

    def __init__(self, i, n):
        self.id = i
        self.name = n

    def __call__(self, *args, **kwargs):
        print('printing args')
        print(*args)

        print('printing kwargs')
        for key, value in kwargs.items():
            print("%s == %s" % (key, value))

e = Employee(10, 'Pankaj')  # creating object
print(callable(e)) 

if callable(e):
    fr_kwg = {'x': 1, 'y': 2}
    e()  # object called as a function, no arguments
    e(10, 20)  # only args
    e.__call__(10, 20)
    e(10, 20, {'x': 1, 'y': 2})  # only args of different types
    e(10, 20, fr_kwg)
    e(10, 20, **fr_kwg)
    e(10, 'A', name='Pankaj', id=20)  # args and kwargs 
 #benefit of __call__
class Employee:
    id = 0
    name = ""
    
    def __init__(self, i, n):
        self.id = i
        self.name = n
        
    def __call__(self, i, n):
        self.id = i
        self.name = n
  
e = Employee(10, 'Pankaj')  # creating object
e(11,'Alie') #you can't update like that coz it isn't callable.once you created object,done.if you wanna create another


 #__str__ and __repr__ 
class Point3D(object): 
    def __init__(self,a=1 , b=1 , c=1):
        self.x = a
        self.y = b
        self.z = c
    '''def __repr__(self):
        return "Point3D(%d, %d, %d)" % (self.x, self.y, self.z)
    def __str__(self):
        return "(%d, %d, %d)" % (self.x, self.y, self.z)'''
my_point = Point3D(1, 2, 3)
# my_point # __repr__ gets called automatically
print (my_point) # __str__ gets called automatically

#difference betweeen repr and str
#str() is used for creating output for end user while repr() is mainly used for debugging and development. 
#repr’s goal is to be unambiguous and str’s is to be readable. For example, if we suspect a float has a small
#rounding error, repr will show us while str may not.
import datetime 

today = datetime.datetime.now()   
# Prints readable format for date-time object 
print (str(today)) 
# prints the official format of date-time object 
print (repr(today))


#__setitem__ and __getitem__
class CustomList(object):
        def __init__(self, elements=0):
            self.my_custom_list = [0] * elements

        def __setitem__(self, index, value):
            self.my_custom_list[index] = value

        def __getitem__(self, index):
            return "Hey you are accessing {} element whose value is: {}".format(index, self.my_custom_list[index])

        def __str__(self):
            return str(self.my_custom_list)

obj = CustomList(12)
obj[0] = 1
print(obj[0])
print(obj)
