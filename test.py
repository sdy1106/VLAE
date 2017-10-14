def build_table():
    bin_table = [""]*1024
    for i in range(0,1024):
        bin_table[i] = '{0:010b}'.format(i)
    return bin_table

a = build_table()
for i in range (100):
    print a[i]
