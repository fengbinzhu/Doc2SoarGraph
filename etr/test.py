def get_index_num_date(tags):
    index=[]
    begin=0
    flag=0
    for i,m in enumerate(tags):
        if flag==0:
            if m==1 or m==2:
                index.append(i)
                flag=1
        if flag==1:
            if m!=1 and m!=2:
                index.append(i)
                flag=0
    return index
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
a='What is the difference in revenue between December 31,2018 to December 31,2019?'
text_tags=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]

words = []
word_tags = []
prev_is_whitespace = True
word2text_idx = []
new_bbox_lists=[]
bbox_index=0
index=get_index_num_date(text_tags)
for i, c in enumerate(a):
    if is_whitespace(c):  # or c in ["-", "–", "~"]:
        prev_is_whitespace = True
        bbox_index+=1
    elif c in ["-", "–", "~","?","!", "$","(",")", "€","£"]:
        words.append(c)
        word_tags.append(text_tags[i]) # use the first char
        word2text_idx.append(i)
        prev_is_whitespace = True
    elif i in index:
        print(i,c)
        words.append(c)
        word_tags.append(text_tags[i]) # use the first char
        word2text_idx.append(i)
        prev_is_whitespace = False
    else:
        if prev_is_whitespace:
            words.append(c)
            word_tags.append(text_tags[i])
            word2text_idx.append(i)

        else:
            words[-1] += c
        prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)
print('1')