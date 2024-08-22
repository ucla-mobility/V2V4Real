from xinshuo_miscellaneous import sort_dict


a = dict()
a['aaa'] = 2
a['sss'] = 1
a['ccc'] = 10

test = sort_dict(a, 'key', 'descending')
print(test)
print(dict(test))
print(test[0:2])