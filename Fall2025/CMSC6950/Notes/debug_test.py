# import pdb


def my_mean(nums):
    it = 0
    sum = 0
    # pdb.set_trace()
    while it < len(nums):
        # breakpoint()
        sum += nums[it]
        it += 1
    return sum/len(nums)


if __name__ == '__main__':
    my_list = [1, 2, 3, 4, 5, 6]
    print(my_mean(my_list))
