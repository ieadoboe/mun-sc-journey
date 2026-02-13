
def mean(data_list):
    try:
        return sum(data_list)/len(data_list)
    except ZeroDivisionError:
        print("Mean of list of length zero is undefined")
        return None
    except TypeError:
        print("Cannot average non-numerical list")
        return None
