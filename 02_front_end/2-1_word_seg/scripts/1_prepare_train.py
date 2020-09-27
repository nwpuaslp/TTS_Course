import codecs
import sys

input_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
output_file = codecs.open(sys.argv[2], 'w', 'utf-8')

for line in input_file.readlines():
    word_list = line.strip().split()
    for word in word_list:
        if len(word) == 1:
            output_file.write(word + "\tS\n")
        else:
            output_file.write(word[0] + "\tB\n")
            for w in word[1:len(word)-1]:
                output_file.write(w + "\tM\n")
            output_file.write(word[len(word)-1] + "\tE\n")
    output_file.write("\n")

input_file.close()
output_file.close()
