import re

str = 'asuuuu@mail.nyuk, hehehe'
match =  re.search(r'([\w.-]+)@([\w.-]+)', str)
if match:
 print(match.group())
 print(match.group(1))
 print(match.group(2))