import csv, sys

input_file = sys.argv[1]
dict_file = sys.argv[2]
norm_file = sys.argv[3]

ifile = open(input_file, 'rb')
dfile = open(dict_file, 'rb')
ofile  = open(norm_file, 'wb')

types = []
try:
    dict = csv.reader(dfile, delimiter='\t')
    for row in dict:
        types.append(row[1])
finally:
    dfile.close()
    
try:
    reader = csv.reader(ifile)
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    titles = reader.next()
    writer.writerow(titles)    
    for row in reader:
        print row
        nrow = [''] * len(row)
        for i in range(0, len(row)):
            if types[i] ==  'category':
                nrow[i] = '0' if row[i] == '1' else '1'
            else:
                nrow[i] = row[i]
        writer.writerow(nrow)
finally:
    ifile.close()
    ofile.close()