import re

input = open("regex_101.txt", "r")
output = open('result.csv', 'w')
output.write('ID,Values'+ '\n')

for text in input:
    match = re.search(r'(\d+):(\d+):(.*)', text)
    if match:
        output.write('"{}","{}"\n'.format(match.group(2), match.group(3)))

output.close()
input.close()