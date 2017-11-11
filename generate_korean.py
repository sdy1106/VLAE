#coding=utf8

# korean
begin = u'\uac00'
end = u'\ud7ff'


begin_num = ord(begin)
end_num = ord(end)


cnt = 0
fw = open('korean.txt', 'w')
for i in range(begin_num, end_num-100, 100):
    if cnt <100:
        fw.write(unichr(i).encode('utf8') + '\n')
    else:
        break
fw.close()

# Japanese

begin = u'\u3040'
end = u'\u30ff'

begin_num = ord(begin)
end_num = ord(end)


cnt = 0
fw = open('Japanese.txt', 'w')
for i in range(begin_num, end_num):
    if i % 2 == 0:
        fw.write(unichr(i).encode('utf8') + '\n')
    cnt += 1
fw.close()