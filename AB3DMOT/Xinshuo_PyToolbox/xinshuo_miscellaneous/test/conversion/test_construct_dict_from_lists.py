from xinshuo_miscellaneous import construct_dict_from_lists

a = [3,5,4,2,2,1]
b = range(len(a))

test = construct_dict_from_lists(b, a)
print(test)