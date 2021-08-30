# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def print_count(dir):
    i = 0
    for file in os.listdir(dir):
        i+=1
    print("一共有图片",i,'张')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_count('train/cats')
    print_count('train/dogs')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
