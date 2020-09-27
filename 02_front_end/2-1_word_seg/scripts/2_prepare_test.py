import codecs
import sys

input_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
output_file = codecs.open(sys.argv[2], 'w', 'utf-8')

for line in input_file.readlines():
    for word in line.strip():
        word = word.strip()
        if word:
            output_file.write(word.strip() + "\n")
    output_file.write("\n")

input_file.close()
output_file.close()
