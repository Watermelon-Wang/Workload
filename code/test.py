import timeit
import pprint

def add():
    return sum(range(111))

# def main():
#     t = timeit.timeit(stmt="add()", number=1)
#     pprint.pprint(t)
#
#
#
#
#
#
# if __name__ == '__main__':
#     main()

t = timeit.timeit(stmt="add()", number=1)
pprint.pprint(t)