
indices = range(0, 32)
filenames = [str(i).zfill(2) + '.txt' for i in indices]

writer = open('data.csv', 'w')

for file in filenames:
    f = open(file, 'r')
    for entry in f:
        nums = entry[entry.index(',') + 2 : entry.index(';')]
        values = [float(num) for num in nums.split(' ')]
        toWrite = ""
        for i, v in enumerate(values):
            toWrite += str(v)
            if i < 2:
                toWrite += ','
            else:
                toWrite += '\n'
        writer.write(toWrite)

