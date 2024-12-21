from collections import Counter, ChainMap, deque
from math import sqrt

from math import floor

if 0:
    def counter_cmp(counter1, counter2):
        return all([counter1[i[0]] >= counter2[i[0]] for i in counter2.items()])


    def minWindow(s: str, t: str) -> str:
        n = len(s)
        start_index, string_counter, substring_counter, return_pair = 0, Counter(), Counter(t), [n, '']
        for end_index in range(n):  ##
            string_counter[s[
                end_index]] += 1  ## increase counter of end index until string include sub string, until substring is contained.
            print(start_index, end_index, string_counter)
            while counter_cmp(string_counter,
                              substring_counter):  # if sub_string is contained, reduce the string, keep the minumal length string containing
                return_pair = min(return_pair, [end_index - start_index + 1,
                                                s[start_index:end_index + 1]])  # minimal length string to return
                string_counter[s[start_index]] -= 1
                start_index += 1  # move start index forward
                print(start_index, end_index, string_counter, return_pair)

        return return_pair[1]


    print(minWindow(s="ADOBECODEBANC", t="ABC"))
# ADOBEC

# import builtins
# pylookup = ChainMap(locals(), globals(), vars(builtins))
# print(pylookup)
# from collections import ChainMap
# dad = {"name": "John", "age": 35}
# mom = {"name": "Jane", "age": 31}
# family = ChainMap(mom, dad)
# family

#
# son = {"name": "Mike", "age": 0}
# family = family.new_child(son)
# for person in family.maps:
#     print(person)
# print(family.parents)
# print(family.values())
from typing import List


def takeout_one(lst, item):
    cpy = lst.copy()
    cpy.remove(item)
    return cpy


def get_perms(word_lst: List[str]):
    if len(word_lst) == 1:
        return word_lst
    elif len(word_lst) == 2:
        return [word_lst[0] + word_lst[1], word_lst[1] + word_lst[0]]
    else:
        w = word_lst.copy()
        w.remove(word_lst[1])

        return [i + item for i in word_lst for item in get_perms(takeout_one(word_lst, i))]


print(get_perms(['ab', 'cd', 'ef']))


def findSubstring(s: str, words: List[str]) -> List[int]:
    import re
    perms = get_perms(words)
    return_lst = [m.start() for item in perms for m in re.finditer(item, s)]
    return sorted(return_lst)


lst = findSubstring("barfoothefoobarman", ["foo", "bar"])
print(lst)
lst2 = findSubstring(s="barfoofoobarthefoobarman", words=["bar", "foo", "the"])
print(lst2)
lst3 = findSubstring(s="aaa", words=["a", "a"])
print(lst2)

import re

text = "Hello world! Hello Python!"
pattern = r"Hello (\w+)"

# findall()
matches = re.findall(pattern, text)
print(matches)
print(re.findall('aa', 'aaa'))
# finditer()
for match in re.finditer('aa', 'aaa'):
    print(match.group(0))  # Hello world
    # print(match.group(1))  # world
    print(match.start())  # 0
    print(match.end())  # 11

from typing import List


def find_all_indexes(string, substring):
    indexes = []
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        indexes.append(start)
        start += 1
    return indexes


def takeout_one(lst, item):
    cpy = lst.copy()
    cpy.remove(item)
    return cpy


def get_perms(word_lst: List[str]):
    if len(word_lst) == 1:
        return word_lst
    elif len(word_lst) == 2:
        return [word_lst[0] + word_lst[1], word_lst[1] + word_lst[0]]
    else:
        w = word_lst.copy()
        w.remove(word_lst[1])

        return [i + item for i in word_lst for item in get_perms(takeout_one(word_lst, i))]


class Solution:

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        import re
        perms = get_perms(words)
        return_lst = [i for item in perms for i in find_all_indexes(s, item)]
        return sorted(list(set(return_lst)))


class Solution_min1:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 0
        if n == 2:
            return 1 if s[0] > s[1] else 0

        return self.minimumSteps(['1'] + s[2:]) + 1 if s[0] > s[1] else self.minimumSteps(s[1:])


class Solution_mini2:
    def minimumSteps(self, s: str) -> int:
        zero_positions = [i for i in range(len(s)) if s[i] == '0']
        print(zero_positions)
        counts = [zero_positions[i] - i for i in range(len(zero_positions))]
        return sum(counts)


# print(Solution_mini2().minimumSteps('101'))

print('Solution_missing')


class Solution_missing:
    def firstMissingPositive(self, nums: List[int]) -> int:
        def swap_elements(lst, i, j):
            lst[i], lst[j] = lst[j], lst[i]

        n = len(nums)

        # Place each positive integer i at index i-1 if possible
        for i in range(n):
            print(i, nums)
            while 0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                print(nums)
                swap_elements(nums, i, nums[i] - 1)
                print(nums)
        print(nums)
        # Find the first missing positive integer
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1

        # If all positive integers from 1 to n are present, return n + 1
        return n + 1

        # n = len(nums)
        #
        # i = 0
        # while i < n and n > 1:
        #
        #     if nums[i] > n or nums[i] < 1:
        #         nums.pop(i)
        #         i = 0
        #         n = len(nums)
        #     else:
        #         i += 1
        # print(nums)
        # n = len(nums)
        # if n == 1:
        #     return 2 if nums[0] == 1 else 1
        #
        # i = 0
        # while i < n:
        #     temp = nums[nums[i]-1]
        #     nums[nums[i]-1] = nums[i]
        #     nums[i] = temp
        #     i += 1
        # # for i in range(n):
        # #     while 0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]:
        # #         swap_elements(nums, i, nums[i] - 1)
        #
        # print(nums)
        # n=len(nums)
        # i=1
        # while i < n:
        #     if nums[i]-nums[i-1] <= 0:
        #         nums.pop(i)
        #         print(i, nums)
        #         n=len(nums)
        #         continue
        #     i +=1
        # n=len(nums)
        # print(nums)
        # if nums[0] > 1:
        #     return 1
        # for i in range(1, n):
        #     if nums[i] - nums[i-1] > 1:
        #         return nums[i-1] + 1
        # return nums[-1] + 1


# s = Solution_missing()
# x = s.firstMissingPositive([1000,3, -11,2, 1,99])
# y = s.firstMissingPositive([350, 2,2,4,0,200,1,3,3,3,4,3])
# print(f'missing = {x}, {y}')


class Solution_fancystring:
    def makeFancyString(self, s: str) -> str:
        n = len(s)
        if n < 3:
            return s
        i = 1
        while i < n:
            while s[i] == s[i - 1] and s[i] == s[i + 1]:
                s.pop(i)
                return self.makeFancyString(s)
            i += 1
        return s


class Solution_subset:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 0:
            return [[]]
        # if len(nums) == 1:
        #     return [nums]
        ret = [[], nums]
        for i in nums:
            nums_copy = nums.copy()
            nums_copy.remove(i)
            if len(nums_copy) > 0 and nums_copy not in ret:
                ret.append(nums_copy)
            subset = self.subsets(nums_copy)
            if len(subset) > 0:
                ret.extend([s for s in subset if s not in ret])
        return ret


s = Solution_subset().subsets([0, 1, 2])
print('subset ', s)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


from typing import Optional


def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    l = ListNode()
    cur = l
    while list1 is not None and list2 is not None:
        if list1.val > list2.val:
            cur.next = list2
            list2 = list2.next
        else:
            cur.next = list1
            list1 = list1.next
        cur = cur.next
    if list1:
        cur.next = list1
    if list2:
        cur.next = list2
    return l.next


list1 = ListNode(1, ListNode(2, ListNode(3)))
list2 = ListNode(1, ListNode(3, ListNode(5)))
lst = mergeTwoLists(list1, list2)
n = lst
while n is not None:
    print(n.val)
    n = n.next


# while n := lst.next is not None:
#     print(n.val)
#     n = n.next
def moveZeroes(nums: List[int]) -> None:
    """
        Do not return anything, modify nums in-place instead.
        """
    i = 0
    popped = 0
    while i < len(nums):
        if nums[i - popped] == 0:
            nums.append(nums.pop(i - popped))
            popped += 1

        i += 1


nums = [0, 0, 0, 2, 3]
# i=0, popped = 0
# 00230, i=1, popped=1
# 02300, i =2, popped=2
# 23000, i =3, popped=3
# 23000, i =4, popped=3
moveZeroes(nums)
print(nums)


def isValid(s: str) -> bool:
    lookup = {"(": ")",
              "[": "]",
              "{": "}"
              }
    reverse_lookup = {")": "(",
                      "]": "[",
                      "}": "{"
                      }

    r = deque()
    for i in range(len(s)):
        l = s[i]
        if l in lookup.keys():
            r.append(l)
        elif l in lookup.values():
            try:
                if r.pop() != reverse_lookup[l]:
                    return False
            except IndexError as e:
                return False
    return len(r) == 0


def maxSubArray(nums: List[int]) -> int:
    res = maxEnding = nums[0]
    idx = []
    for i in range(1, len(nums)):

        # Find the maximum sum ending at index i by either extending
        # the maximum sum subarray ending at index i - 1 or by
        # starting a new subarray from index i
        if maxEnding < 0:
            idx = [i]
        else:
            idx.append(i)
        maxEnding = max(maxEnding + nums[i], nums[i])

        # Update res if maximum subarray sum ending at index i > res
        res = max(res, maxEnding)
        if res < maxEnding:
            idx.pop()

    return res, [nums[j] for j in idx]


print(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))


def longestDupSubstring(s: str) -> str:
    def get_substrings(st: str, n: int) -> list[str]:
        return [st[i:n + i] for i in range(len(st))]

    m = len(s)
    for j in range(m - 1, 0, -1):
        subs = get_substrings(s, j)
        counts = Counter(subs)
        for i, c in counts.items():
            if c > 1:
                return i
    return ""


# print(longestDupSubstring("shabhlesyffuflsdxvvvoiqfjacpacgoucvrlshauspzdrmdfsvzinwdfmwrapbzuyrlvulpalqltqshaskpsoiispneszlcgzvygeltuctslqtzeyrkfeyohutbpufxigoeagvrfgkpefythszpzpxjwgklrdbypyarepdeskotoolwnmeibkqpiuvktejvbejgjptzfjpfbjgkorvgzipnjazzvpsjxjscekiqlcqeawsdydpsuqewszlpkgkrtlwxgozdqvyynlcxgnskjjmdhabqxbnnbflsscammppnlwyzycidzbhllvfvheujhnxrfujwmhwiamqplygaujruuptfdjmdqdndyzrmowhctnvxryxtvzzecmeqdfppsjczqtyxlvqwafjozrtnbvshvxshpetqijlzwgevdpwdkycmpsehxtwzxcpzwyxmpawwrddvcbgbgyrldmbeignsotjhgajqhgrttwjesrzxhvtetifyxwiyydzxdqvokkvfbrfihslgmvqrvvqfptdqhqnzujeiilfyxuehhvwamdkkvfllvdjsldijzkjvloojspdbnslxunkujnfbacgcuaiohdytbnqlqmhavajcldohdiirxfahbrgmqerkcrbqidstemvngasvxzdjjqkwixdlkkrewaszqnyiulnwaxfdbyianmcaaoxiyrshxumtggkcrydngowfjijxqczvnvpkiijksvridusfeasawkndjpsxwxaoiydusqwkaqrjgkkzhkpvlbuqbzvpewzodmxkzetnlttdypdxrqgcpmqcsgohyrsrlqctgxzlummuobadnpbxjndtofuihfjedkzakhvixkejjxffbktghzudqmarvmhmthjhqbxwnoexqrovxolfkxdizsdslenejkypyzteigpzjpzkdqfkqtsbbpnlmcjcveartpmmzwtpumbwhcgihjkdjdwlfhfopibwjjsikyqawyvnbfbfaikycrawcbkdhnbwnhyxnddxxctwlywjcisgqfsctzatdgqqauuvgclicdrpjcphysqdjaflpdbmvnhqggixxzcmpsysbwfkzwxzjictnngufpqhcxlbkodyrqlfomlkiefbmcfenugzqnyqqvgpxonmizkpjdlaqyyowjagzkzrzvcrupfyofeftyfvoqorzvxphhdhydnqiyiczfcgzsecxzsoaobwrixcajabjnvtoerzwayjowahrmuixmmkbtchogfizmvbjnpespxngxjxntohzatlpkcmpphmewevpteharnszafbpbexrvnbedieojezdhnyooiivhnhakilvkobxepbksnqrtxxuqhalvtjspyvporalbliiwjciamlhttaydhxoelimuorjnfvebjhcocbkrgbguwdncodskzzoqrzgavsbjcippetltqaxjhkqacwlgmsbxezqubyzeznnsoqegkykzlxohvitbmjcxllbrvgdijyovpjyeaojlyxqwnheyblznwoyikhqiutotpfukyqkvatxotulvlqzfcvskdccuixthzqrwymzccosjmjqjigehcnfphjuuybaxxukconatzseljyirycbhucxmwwftulfwfmyqyprlnsmxzyfmgjctgeunuuexhbrbsaaingqxqrjvpuhbvcmyztmkgenhonajrkzfrqjinjrbmjyinhwvlcmmxvbgvjgfmaoliosmxbonvlzoiqvkxxtoposygcgkcotohcrauivxxvmrghuauadwojxjligrgstczirnvhqpzwgjbvqzlkxltqnqrfxieggnuriytavbnouwhuamdtlspednyckixkhxedjmotiuucewllthhducwgwmgzxsbkqzfnqfynwisvsctyqdoaiypjivtxkxgoyhwhccklbdjoqykaqzljejlizgbehekmkfetvgfstmypmfnyoundudqlorcogbzoznddfalthwpmiewkmvogmzirbprjftbtffjrkrfminnechitfyfaujgtugadqbrskulsjbaunonxolauvsifevpdyurvfocxtkizflcuvltzuhwyhlbxaphifhtgkfktfnnmocpenrlujsuppbbuorvtubuiyszawzftijwhwgdyubjmmodzybiyunksuixnkariubegpdgctbayaynfskkuyhjvegsjwsbppodvhpjdjlzhxixswdncapxyfjspxeqxdfkhockvrzoisikaymoiqzqbjyoscwegfomlnurwboesfiszetebjblaolnovgvfcpnbemwambkhwcgdbhvkoluhjfxlfrfaeedocdilaboesauplmttewlbojkocklhsbzrtzeyhqtmgroupbzlymupmupsvlkzchclujuozzmngjvktzstsvocxrziuxelruwojzaleyrkjkdleavwqxwgjdbtiywqtdtaamrlcjogxufhgvoqpqkgopbtyqchzhexqgecheahjzxapqjdylzjqhzlzssbjmokncxalgasexztnlzfisxxpeerywlrjdohprewwnlwdbtwmfnnxnoolfhqqxzcvoymdbvmaoliedpvwzyvgyrwjguvoqnxrnaeqwvcfrqkwjmlvxovptultyfuxerelpfgctnpdxluqeruxkxqntosggfjqmrnlnkhhilznpycdrnemnktcsmzufpqgiraphzmgfhevzejhavsypohpttnnowfahpxfwmvxgwfuomxemdkzdlzldesmowzmhwoydnsovwykxqyllbmcurlvtwcfwxvvkxfknwwcwfjkzjtonalgijdsulcfagehiptszrcltbbypopdbmdfkyofelmrdmdbceguyxnkheqqtbletpqmjugpckmjyuuvsbqhyzmposwcgscnishluuhnwkyrkujefpgtsqrmcoortgitpdoagdncxlofkqozgngbtmlyyoyodcmcwetdtltupjrtemrjswekkfjvfecmvagyptjjuwsqpjwlxxosqhpssdvjraaicjfwvesyqfbumjjbqytkinpldxopxjzmvpigmberobyzyxwvwmlmbziduqhmbescgkvhgqtalmaxfsjlysmvrizgvrudstiwmaahtqehfbofvqwgqygvseykmgmhgjbxcrtdjqvojvyhohallyewqelzhjeuqmmsqhkluvqsfmxzbqqokehfoqrlqmwpnwojfowqpqebnuggeuvsszgfywceolvabyvbrwatuyherijsdqvpyyhdyradbammmchqkvdbxpbrxzrpfrsiiezvowrfqejibvociujtcwbygvfwojgfnvvwqlqqgipxhrogppzghtnweodaxuqxknnqnajlnsvheiycsvifvoljsncgnunsqcymnyoeeslrjflpprvtksimffvnuvakskdakvmlkpowfpfzdrcfctikhvvbagrvjlzjydnlmspzyynyjjfxnozpjjgjelipswsmfroitqphzsuqgumlnkxksbzhrsvcnfwufofhurmhksvvfjzggbtgrezkrkqmhduyqgwuwxoxaiifemtwrbilftiuhcgpjvqxldrnlzphdffncevlcyrxlpbwuswjfdegexeoooshdfqtqithpfocyowaqeedikssptyvkabhtaeotcwxccgguuotqvypugpcbwzalxwqbjdcokoxjnqhggpbbfeyjiellsikiqqtxpvzmjsfleepjpbxpeicxfcwbpprzgcrjgjaxshewradetsqsvfmcxptmksecfpynqzpctqpogcgokzrkltsbmwxkmynasohpkzjupapngusnvdjfqezqhyikllgkelewwwhhbdjvxdagnnxscjkotbbmhzkqbjwuwidrnvmztikmqjcxmcpgkoudhydmdvberfuvjnhlnfcsbpzmuquvrgogtfwefhqzkmxxgadtvjpxvurxprbsssihviypclwkjfaatzjxtvlzwaacqlwnqetgkldqaqghuihrgxbbpmjfsvaigqrhiiskkfibaeilqptkdsqqfwxeixuxgkiboaqnuaeutjcydnxyxnmattjrrxmthwvyipgazaxgrrjcvdnyxpktsldhluhicyqprxhljyfhawuvoonrwyklcdlmdvsgqrwqqomisksftsfyeifmupvylkjbagzyctuifbsrugqsbrkvskmundmczltpamhmgqespzgrkxebsvubrlmkwyqhjyljnkeqvdxtjxjvzlrubsiiahciwefwsswgssxmvyvgjrobvubcbgjomqajmotbcgqjudneovfbjtjzwqtsovzshmxeqofssukkvcdwlsdtyplrlgwtehnwvhhegtwkwnqqdiajpcaajsylesadaiflruewhrbrogbujbppunsqgytgnyuhnkejhccavaptbydtqhvyatftxcaaljyhhkkadzdhhzawgndunwwgknnbtqaddpszqgummmnomfqmdxqtwjexsbadfdqhnyixjslsaisscocbabivzokkgiinqqzsrtfpzjmxfkqmuzzlelqjtjidjarkwbwlcqrefokrlwdmuzyffdtajnqoimlzzpcgpjjwlqkusefzbgznhexzojxnzxmmedobgvdabtdoiskozrdrjscxwivaekrkyyfynuktmgyziteavdxfctvkfkrmsdwpaywzbkeojeycwdkectydojttisizruilwokhepscqdnjygiakomkhyujaffxjyxqmvkemqihpcdygprdeaxgjbkonfvgtzayfbmgwsskoyxjlknwwtehhhpjllhkcblyaxnbekoidbbyqvdqqsyfcemylmqskpxifcnhmemkkitqtbfwhmyemkzightkjbhlquticivpeeclpamsqoztxvdtcqbsonxyecnhcadtghkjckhrcdfggnqlwurydzbeybqkcfnnbwkciwaqdzgmcrbltvcuftxsqfpxnoombsbeoqxivgtkrjklxatgcorkdrvmngwlekeopgecefzxtprcoajoopxviijxilxfiwuajsbtcctfcqqgzhyjmonwdbyjlnneidyaqhhothzpzaxcthvbxpdcqofaeamxbqjwhunnqwclhcqhagawjsxygorgbuqryzickyplbaivkabbrkibqzqacabbwmnpndaqvbknbqcjuywxjrdbznndomwbbqfgulwczydrhrocebhygriiwxmwtjjyqqiqrjblxuamddlsiocdoysdaacuovljtpgocephnxuugdyggsnhcqiqhulyhlxiwzvhrtbmvjnhacwunckqzhpcthqgsupptdwkfeprbg"))

def longestDupSubstringSlidingWindow(s: str) -> str:
    """
            This problem can be solved using Sliding Window Technique.
            Logic:
                1. Iterate over the string from 0th index
                2. For each index, define a window of 1 initially.
                3. Check for the existence of the window in the remaining string:
                    a. If found, increase the size of window by 1 and repeat.
                    b. Else Goto next index. For next index, the size window will not start by 1 again as we have already found for 1. So for every next index, size of window will start from the size at previous index to avoid checking for repeating size.
        """
    result = ""
    right_index = 1
    for left_index in range(len(s)):
        slide_window = s[left_index:left_index + right_index]
        right_segment = s[left_index + 1:]

        while slide_window in right_segment:
            result = slide_window
            right_index += 1
            slide_window = s[left_index:left_index + right_index]

    return result

    # print(left_index, right_index, result, slide_window)


print(longestDupSubstringSlidingWindow(
    "shabhlesyffuflsdxvvvoiqfjacpacgoucvrlshauspzdrmdfsvzinwdfmwrapbzuyrlvulpalqltqshaskpsoiispneszlcgzvygeltuctslqtzeyrkfeyohutbpufxigoeagvrfgkpefythszpzpxjwgklrdbypyarepdeskotoolwnmeibkqpiuvktejvbejgjptzfjpfbjgkorvgzipnjazzvpsjxjscekiqlcqeawsdydpsuqewszlpkgkrtlwxgozdqvyynlcxgnskjjmdhabqxbnnbflsscammppnlwyzycidzbhllvfvheujhnxrfujwmhwiamqplygaujruuptfdjmdqdndyzrmowhctnvxryxtvzzecmeqdfppsjczqtyxlvqwafjozrtnbvshvxshpetqijlzwgevdpwdkycmpsehxtwzxcpzwyxmpawwrddvcbgbgyrldmbeignsotjhgajqhgrttwjesrzxhvtetifyxwiyydzxdqvokkvfbrfihslgmvqrvvqfptdqhqnzujeiilfyxuehhvwamdkkvfllvdjsldijzkjvloojspdbnslxunkujnfbacgcuaiohdytbnqlqmhavajcldohdiirxfahbrgmqerkcrbqidstemvngasvxzdjjqkwixdlkkrewaszqnyiulnwaxfdbyianmcaaoxiyrshxumtggkcrydngowfjijxqczvnvpkiijksvridusfeasawkndjpsxwxaoiydusqwkaqrjgkkzhkpvlbuqbzvpewzodmxkzetnlttdypdxrqgcpmqcsgohyrsrlqctgxzlummuobadnpbxjndtofuihfjedkzakhvixkejjxffbktghzudqmarvmhmthjhqbxwnoexqrovxolfkxdizsdslenejkypyzteigpzjpzkdqfkqtsbbpnlmcjcveartpmmzwtpumbwhcgihjkdjdwlfhfopibwjjsikyqawyvnbfbfaikycrawcbkdhnbwnhyxnddxxctwlywjcisgqfsctzatdgqqauuvgclicdrpjcphysqdjaflpdbmvnhqggixxzcmpsysbwfkzwxzjictnngufpqhcxlbkodyrqlfomlkiefbmcfenugzqnyqqvgpxonmizkpjdlaqyyowjagzkzrzvcrupfyofeftyfvoqorzvxphhdhydnqiyiczfcgzsecxzsoaobwrixcajabjnvtoerzwayjowahrmuixmmkbtchogfizmvbjnpespxngxjxntohzatlpkcmpphmewevpteharnszafbpbexrvnbedieojezdhnyooiivhnhakilvkobxepbksnqrtxxuqhalvtjspyvporalbliiwjciamlhttaydhxoelimuorjnfvebjhcocbkrgbguwdncodskzzoqrzgavsbjcippetltqaxjhkqacwlgmsbxezqubyzeznnsoqegkykzlxohvitbmjcxllbrvgdijyovpjyeaojlyxqwnheyblznwoyikhqiutotpfukyqkvatxotulvlqzfcvskdccuixthzqrwymzccosjmjqjigehcnfphjuuybaxxukconatzseljyirycbhucxmwwftulfwfmyqyprlnsmxzyfmgjctgeunuuexhbrbsaaingqxqrjvpuhbvcmyztmkgenhonajrkzfrqjinjrbmjyinhwvlcmmxvbgvjgfmaoliosmxbonvlzoiqvkxxtoposygcgkcotohcrauivxxvmrghuauadwojxjligrgstczirnvhqpzwgjbvqzlkxltqnqrfxieggnuriytavbnouwhuamdtlspednyckixkhxedjmotiuucewllthhducwgwmgzxsbkqzfnqfynwisvsctyqdoaiypjivtxkxgoyhwhccklbdjoqykaqzljejlizgbehekmkfetvgfstmypmfnyoundudqlorcogbzoznddfalthwpmiewkmvogmzirbprjftbtffjrkrfminnechitfyfaujgtugadqbrskulsjbaunonxolauvsifevpdyurvfocxtkizflcuvltzuhwyhlbxaphifhtgkfktfnnmocpenrlujsuppbbuorvtubuiyszawzftijwhwgdyubjmmodzybiyunksuixnkariubegpdgctbayaynfskkuyhjvegsjwsbppodvhpjdjlzhxixswdncapxyfjspxeqxdfkhockvrzoisikaymoiqzqbjyoscwegfomlnurwboesfiszetebjblaolnovgvfcpnbemwambkhwcgdbhvkoluhjfxlfrfaeedocdilaboesauplmttewlbojkocklhsbzrtzeyhqtmgroupbzlymupmupsvlkzchclujuozzmngjvktzstsvocxrziuxelruwojzaleyrkjkdleavwqxwgjdbtiywqtdtaamrlcjogxufhgvoqpqkgopbtyqchzhexqgecheahjzxapqjdylzjqhzlzssbjmokncxalgasexztnlzfisxxpeerywlrjdohprewwnlwdbtwmfnnxnoolfhqqxzcvoymdbvmaoliedpvwzyvgyrwjguvoqnxrnaeqwvcfrqkwjmlvxovptultyfuxerelpfgctnpdxluqeruxkxqntosggfjqmrnlnkhhilznpycdrnemnktcsmzufpqgiraphzmgfhevzejhavsypohpttnnowfahpxfwmvxgwfuomxemdkzdlzldesmowzmhwoydnsovwykxqyllbmcurlvtwcfwxvvkxfknwwcwfjkzjtonalgijdsulcfagehiptszrcltbbypopdbmdfkyofelmrdmdbceguyxnkheqqtbletpqmjugpckmjyuuvsbqhyzmposwcgscnishluuhnwkyrkujefpgtsqrmcoortgitpdoagdncxlofkqozgngbtmlyyoyodcmcwetdtltupjrtemrjswekkfjvfecmvagyptjjuwsqpjwlxxosqhpssdvjraaicjfwvesyqfbumjjbqytkinpldxopxjzmvpigmberobyzyxwvwmlmbziduqhmbescgkvhgqtalmaxfsjlysmvrizgvrudstiwmaahtqehfbofvqwgqygvseykmgmhgjbxcrtdjqvojvyhohallyewqelzhjeuqmmsqhkluvqsfmxzbqqokehfoqrlqmwpnwojfowqpqebnuggeuvsszgfywceolvabyvbrwatuyherijsdqvpyyhdyradbammmchqkvdbxpbrxzrpfrsiiezvowrfqejibvociujtcwbygvfwojgfnvvwqlqqgipxhrogppzghtnweodaxuqxknnqnajlnsvheiycsvifvoljsncgnunsqcymnyoeeslrjflpprvtksimffvnuvakskdakvmlkpowfpfzdrcfctikhvvbagrvjlzjydnlmspzyynyjjfxnozpjjgjelipswsmfroitqphzsuqgumlnkxksbzhrsvcnfwufofhurmhksvvfjzggbtgrezkrkqmhduyqgwuwxoxaiifemtwrbilftiuhcgpjvqxldrnlzphdffncevlcyrxlpbwuswjfdegexeoooshdfqtqithpfocyowaqeedikssptyvkabhtaeotcwxccgguuotqvypugpcbwzalxwqbjdcokoxjnqhggpbbfeyjiellsikiqqtxpvzmjsfleepjpbxpeicxfcwbpprzgcrjgjaxshewradetsqsvfmcxptmksecfpynqzpctqpogcgokzrkltsbmwxkmynasohpkzjupapngusnvdjfqezqhyikllgkelewwwhhbdjvxdagnnxscjkotbbmhzkqbjwuwidrnvmztikmqjcxmcpgkoudhydmdvberfuvjnhlnfcsbpzmuquvrgogtfwefhqzkmxxgadtvjpxvurxprbsssihviypclwkjfaatzjxtvlzwaacqlwnqetgkldqaqghuihrgxbbpmjfsvaigqrhiiskkfibaeilqptkdsqqfwxeixuxgkiboaqnuaeutjcydnxyxnmattjrrxmthwvyipgazaxgrrjcvdnyxpktsldhluhicyqprxhljyfhawuvoonrwyklcdlmdvsgqrwqqomisksftsfyeifmupvylkjbagzyctuifbsrugqsbrkvskmundmczltpamhmgqespzgrkxebsvubrlmkwyqhjyljnkeqvdxtjxjvzlrubsiiahciwefwsswgssxmvyvgjrobvubcbgjomqajmotbcgqjudneovfbjtjzwqtsovzshmxeqofssukkvcdwlsdtyplrlgwtehnwvhhegtwkwnqqdiajpcaajsylesadaiflruewhrbrogbujbppunsqgytgnyuhnkejhccavaptbydtqhvyatftxcaaljyhhkkadzdhhzawgndunwwgknnbtqaddpszqgummmnomfqmdxqtwjexsbadfdqhnyixjslsaisscocbabivzokkgiinqqzsrtfpzjmxfkqmuzzlelqjtjidjarkwbwlcqrefokrlwdmuzyffdtajnqoimlzzpcgpjjwlqkusefzbgznhexzojxnzxmmedobgvdabtdoiskozrdrjscxwivaekrkyyfynuktmgyziteavdxfctvkfkrmsdwpaywzbkeojeycwdkectydojttisizruilwokhepscqdnjygiakomkhyujaffxjyxqmvkemqihpcdygprdeaxgjbkonfvgtzayfbmgwsskoyxjlknwwtehhhpjllhkcblyaxnbekoidbbyqvdqqsyfcemylmqskpxifcnhmemkkitqtbfwhmyemkzightkjbhlquticivpeeclpamsqoztxvdtcqbsonxyecnhcadtghkjckhrcdfggnqlwurydzbeybqkcfnnbwkciwaqdzgmcrbltvcuftxsqfpxnoombsbeoqxivgtkrjklxatgcorkdrvmngwlekeopgecefzxtprcoajoopxviijxilxfiwuajsbtcctfcqqgzhyjmonwdbyjlnneidyaqhhothzpzaxcthvbxpdcqofaeamxbqjwhunnqwclhcqhagawjsxygorgbuqryzickyplbaivkabbrkibqzqacabbwmnpndaqvbknbqcjuywxjrdbznndomwbbqfgulwczydrhrocebhygriiwxmwtjjyqqiqrjblxuamddlsiocdoysdaacuovljtpgocephnxuugdyggsnhcqiqhulyhlxiwzvhrtbmvjnhacwunckqzhpcthqgsupptdwkfeprbg"))


def summaryRanges(nums: List[int]) -> List[str]:
    l, r = 0, 1
    result = []

    n = len(nums)
    while l < n:
        arr = [nums[l]]
        while r < n and nums[r] == arr[-1] + 1:
            arr.append(nums[r])
            r += 1
        l = r
        r = l + 1
        result.append(arr)
    return [f"{a[0]}->{a[-1]}" if len(a) > 1 else f"{a[0]}" for a in result]


print(summaryRanges([0, 1, 2, 4, 5, 7]))


def wordPattern(pattern: str, s: str) -> bool:
    arr = s.split(' ')
    if len(set(pattern)) != len(set(arr)):
        return False
    d = {}
    for i in range(len(pattern)):
        if pattern[i] in d.keys() and d[pattern[i]] != arr[i]:
            return False
        d[pattern[i]] = arr[i]

    return True


print(wordPattern("abba", "dog dog dog dog"))


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def oddEvenList(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    o_node = head
    e_node = head.next
    e_head = e_node
    while e_node and e_node.next:
        o_node.next = e_node
        o_node = o_node.next
        e_node.next = o_node.next
        e_node = e_node.next
    o_node.next = e_head
    return head


def print_linkedlist(l: List[ListNode]):
    while l:
        print(l.val)
        l = l.next


h = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))

# print_linkedlist(oddEvenList(h))


cube = lambda x: x ** 3  # complete the lambda function


# def fib(n):
#     if n in [0,1]:
#         return n
#
#     return fib(n-1) + fib(n-2)

def fibonacci(n):
    if n == 0:
        return [0]
    if n == 1:
        return [0, 1]
    last = fibonacci(n - 1)
    return last + [last[-1] + last[-2]]

    # return a list of fibonacci numbers


n = 5
print(fibonacci(n))
print(list(map(cube, fibonacci(n))))


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def to_number(l: Optional[ListNode]) -> int:
    n = 0
    i = 0
    while l:
        n += l.val * 10 ** i
        i += 1
        l = l.next
    return n


def to_list(num):
    digits = list(map(int, str(num)))
    h = ListNode(digits[-1])
    c = h
    for d in digits[-1::-1]:
        c.next = ListNode(d)
        c = c.next
    return h


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    n1 = to_number(l1)
    n2 = to_number(l2)
    n = n1 + n2
    return to_list(n)


def print_list(l):
    while l:
        print(l.val)
        l = l.next


# print(to_number(ListNode(1, ListNode(2))))
# print_list(to_list(321))
# print_list(addTwoNumbers(ListNode(1, ListNode(2)), ListNode(2, ListNode(1))))
def find_divisors(num: int) -> List[int]:
    res = [1]
    u = floor(sqrt(num)) + 1

    for i in range(2, u):
        if num % i == 0:
            res.extend([i, int(num / i)])
    res.append(num)
    return res


def sumFourDivisors(nums):
    sums = 0
    for item in nums:
        divs = find_divisors(item)
        if len(divs) == 4:
            sums += sum(divs)
    return sums


print(find_divisors(4))
print(sumFourDivisors([21, 7, 4, 1]))


def find_prime_masks(n):
    N = [0] * n
    i = 2
    for i in range(2, 4):
        if N[i]: continue
        N[i * i:n:i] = [1] * ((n - 1) // i - i + 1)
    return N


mask = find_prime_masks(100)
primes = [i for i in range(2, 100) if mask[i] == 0]
print(find_prime_masks(100))
print(primes)


def products(a: List[int]) -> int:
    prod = 1
    for i in range(len(a)):
        prod = prod * a[i]
    return prod


def split_list(a: List[int]) -> List[List[int]]:
    if len(a) == 1:
        return [a]
    if len(a) == 2:
        return [a, a[:-1], a[1:]]
    return a + split_list(a[:-1]) + split_list(a[1:])


def numSubarrayProductLessThanK(nums: List[int], k: int) -> int:
    n, res = len(nums), 0
    for i in range(n):
        # if nums[i] < k:
        #     res += 1
        #     print(f'i={i}, res={res}, {[nums[i]]}')
        prod = nums[i]
        for j in range(i, n):
            print(f'i={i}, j={j} res={res}, prod={prod} {nums[i:j + 1]}')
            if prod < k:
                res += 1
                prod *= nums[j + 1] if j < n - 1 else 1
            else:
                break
    return res


# print(products([1,2,3]))
# print(split_list([1,2,3,4]))
print(numSubarrayProductLessThanK([1, 2, 3, 4], 100))


def topKFrequent(nums: List[int], k: int) -> List[int]:
    counter = Counter(nums)
    ordered = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
    return list(ordered.keys())[:k]


# print(topKFrequent([1,1,1,2,2,3], 2))

class Solution0:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        subs = [arr[i:j + 1] for i in range(len(arr)) for j in range(i, len(arr))]
        return sum([min(item) for item in subs])


class Solution2:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        s = 0
        for i in range(n):
            for j in range(i, n):
                s += min(arr[i:j + 1])
        return s


def findMaxAverage(nums: List[int], k: int) -> float:
    n = len(nums)
    if n <= k:
        return sum(nums) / k
    if k == 1:
        return max(nums) / k
    accum_delta = 0  # nums[k] - nums[0]
    max_index, max_sum = 0, sum(nums[0:k])
    for i in range(1, n - k + 1):  # (0,1)
        delta = nums[i + k - 1] - nums[i - 1]
        accum_delta += delta

        if accum_delta > 0 and delta > 0:
            max_index = i
            accum_delta = 0
            # print(i, nums[i], nums[i+k-1], accum_delta)

    print(max_index, max_sum)
    return sum(nums[max_index:max_index + k]) / k


print(findMaxAverage([9, 7, 3, 5, 6, 2, 0, 8, 1, 9], 6))
print(findMaxAverage([1, 12, -5, -6, 50, 3], 4))

x = [64, 514, 188, 145, 614, 214, 888, 344, 380, 781, 346, 668, 165, 687, 236, 551, 338, 609, 401, 118, 662, 911, 907,
     792, 83, 942, 213, 427, 442, 979, 700, 507, 493, 888, 4, 107, 454, 893, 803, 834, 26, 150, 502, 191, 189, 738, 742,
     528, 699, 496, 998, 361, 407, 257, 154, 842, 551, 719, 270, 994, 699, 970, 853, 544, 210, 209, 652, 665, 102, 807,
     499, 480, 957, 354, 672, 499, 92, 766, 27, 792, 614, 377, 505, 373, 986, 11, 216, 889, 731, 838, 883, 782, 160, 88,
     326, 370, 298, 330, 35, 752, 138, 887, 233, 447, 241, 257, 946, 685, 23, 325, 829, 638, 54, 335, 11, 392, 346, 579,
     282, 429, 417, 165, 211, 577, 606, 890, 300, 904, 572, 335, 656, 710, 222, 889, 510, 815, 498, 808, 501, 522, 486,
     330, 512, 540, 665, 875, 933, 364, 455, 567, 793, 224, 732, 357, 154, 690, 599, 454, 594, 171, 141, 603, 234, 716,
     844, 744, 531, 343, 904, 32, 865, 390, 715, 729, 931, 380, 604, 216, 96, 411, 783, 890, 636, 867, 599, 790, 558,
     198, 596, 504, 369, 737, 107, 603, 805, 952, 699, 337, 295, 604, 721, 512, 994, 436, 241, 277, 169, 197, 845, 265,
     609, 628, 507, 245, 848, 106, 387, 758, 656, 335, 262, 26, 72, 370, 981, 230, 674, 681, 567, 969, 637, 640, 833,
     983, 429, 426, 261, 598, 623, 106, 215, 232, 87, 723, 829, 935, 181, 568, 693, 838, 903, 307, 216, 328, 677, 197,
     558, 703, 230, 477, 24, 219, 469, 857, 203, 898, 283, 464, 848, 259, 922, 64, 843, 9, 139, 673, 296, 320, 593, 341,
     510, 849, 649, 726, 177, 678, 276, 87, 382, 858, 916, 758, 78, 385, 616, 281, 284, 251, 97, 132, 862, 371, 548,
     706, 733, 687, 379, 29, 360, 324, 723, 870, 173, 372, 949, 702, 402, 225, 141, 136, 83, 57, 895, 513, 443, 863,
     794, 79, 466, 243, 211, 329, 615, 112, 35, 700, 799, 766, 729, 159, 90, 452, 382, 616, 176, 331, 318, 579, 908,
     812, 67, 343, 869, 962, 857, 664, 825, 3, 743, 292, 247, 307, 621, 214, 419, 656, 914, 570, 422, 995, 82, 864, 800,
     464, 480, 328, 147, 151, 259, 407, 315, 327, 750, 184, 641, 607, 201, 467, 611, 944, 111, 210, 603, 84, 776, 374,
     92, 42, 945, 866, 389, 379, 730, 541, 195, 563, 870, 694, 66, 481, 101, 381, 808, 851, 917, 450, 811, 118, 269,
     774, 415, 732, 336, 370, 816, 112, 745, 908, 506, 42, 126, 247, 773, 208, 789, 968, 771, 11, 662, 189, 492, 763,
     922, 301, 966, 840, 103, 777, 310, 372, 551, 77, 104, 887, 448, 272, 351, 545, 532, 209, 939, 658, 457, 712, 866,
     246, 32, 990, 257, 694, 531, 749, 809, 454, 402, 127, 646, 505, 905, 956, 229, 808, 386, 685, 696, 834, 957, 399,
     731, 841, 609, 670, 499, 66, 734, 718, 664, 766, 60, 273, 812, 591, 22, 621, 45, 425, 100, 691, 282, 5, 0, 512,
     814, 386, 197, 862, 572, 507, 613, 303, 348, 574, 325, 200, 992, 59, 918, 656, 825, 978, 929, 637, 921, 304, 610,
     967, 81, 710, 10, 363, 68, 362, 227, 234, 748, 777, 448, 672, 284, 61, 975, 984, 636, 300, 184, 628, 711, 102, 637,
     536, 432, 918, 525, 354, 222, 135, 673, 303, 198, 35, 19, 618, 398, 598, 852, 498, 375, 300, 171, 11, 713, 498,
     996, 701, 151, 180, 682, 862, 635, 671, 751, 419, 589, 276, 125, 812, 764, 150, 467, 962, 186, 838, 580, 936, 437,
     784, 434, 164, 84, 957, 176, 149, 456, 172, 203, 959, 704, 885, 821, 691, 556, 572, 111, 145, 201, 588, 309, 965,
     739, 129, 279, 925, 967, 859, 213, 756, 995, 999, 921, 79, 957, 97, 580, 765, 621, 783, 724, 325, 668, 897, 17,
     576, 822, 480, 74, 23, 68, 383, 988, 807, 512, 267, 84, 832, 478, 297, 588, 473, 297, 509, 904, 606, 958, 484, 723,
     579, 620, 799, 905, 640, 48, 274, 217, 870, 754, 643, 893, 174, 26, 233, 982, 891, 852, 418, 75, 330, 716, 663,
     155, 365, 525, 59, 323, 483, 896, 46, 63, 868, 845, 320, 508, 245, 594, 77, 116, 700, 720, 361, 874, 99, 947, 208,
     990, 799, 627, 65, 482, 695, 80, 637, 60, 605, 49, 735, 89, 297, 781, 152, 165, 978, 824, 25, 223, 770, 103, 691,
     470, 823, 53, 696, 922, 352, 905, 264, 151, 884, 681, 985, 579, 762, 975, 991, 367, 376, 726, 808, 673, 859, 960,
     190, 837, 136, 215, 412, 906, 318, 456, 728, 494, 861, 425, 416, 213, 682, 681, 364, 566, 362, 702, 145, 476, 677,
     136, 844, 53, 214, 652, 78, 73, 965, 268, 262, 101, 835, 674, 360, 154, 130, 88, 0, 991, 865, 416, 556, 547, 449,
     273, 465, 164, 975, 962, 640, 652, 450, 484, 57, 664, 489, 135, 737, 806, 755, 999, 259, 590, 26, 619, 96, 508, 60,
     96, 852, 925, 865, 408, 825, 314, 681, 290, 478, 656, 605, 471, 660, 55, 307, 717, 72, 796, 204, 161, 954, 311,
     161, 566, 254, 539, 185, 702, 47, 597, 799, 251, 523, 16, 660, 700, 682, 693, 990, 513, 702, 947, 984, 362, 3, 291,
     432, 75, 440, 988, 236, 394, 300, 749, 312, 554, 640, 850, 256, 40, 447, 407, 291, 322, 775, 303, 22, 458, 997,
     365, 971, 699, 312, 307, 413, 315, 598, 197, 742, 390, 186, 331, 137, 838, 432, 449, 744, 73, 299, 352, 113, 747,
     112, 404, 69, 887, 60, 444, 697, 57, 809, 668, 108, 121, 975, 873, 789, 926, 71, 883, 316, 609, 214, 453, 799, 999,
     903, 895, 72, 202, 247, 537, 301, 711, 293, 723, 599, 353, 167, 648, 410, 976, 669, 870, 449, 996, 744, 238, 922,
     167, 122, 591, 776, 688, 44, 927, 687, 299, 822, 111, 854, 69, 648, 155, 133, 294, 230, 732, 647]
print(sum(x[601:999]) / 398, 195830 / 398)
print(findMaxAverage(x, 398))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    return inorderTraversal(root.left) + ([root.val] if root.val is not None else []) + inorderTraversal(root.right)
tree = TreeNode(1, right=TreeNode(2, left=TreeNode(3)))
print(inorderTraversal(tree))
# print(inorderTraversal(tree.left))
#


def subArrayRanges(nums: List[int]) -> int:
    n = len(nums)
    print(nums)
    if n < 2 : return 0
    if n == 2:
        # print(nums[1], nums[0])
        return abs(nums[1] - nums[0])
    rng = max(nums)-min(nums)
    # print(max(nums) - min(nums))
    return rng + subArrayRanges(nums[:n-1]) + subArrayRanges(nums[1:n])

print(subArrayRanges([4, -2, -3, 4,1])) # 6+1+7+3, 7+7+7,7+7, 7

def missingNumber(nums: List[int]) -> int:

    missing = set(range(len(nums) + 1)) - set(nums)
    return missing.pop()

print(missingNumber([1]))

def licenseKeyFormatting(s: str, k: int) -> str:
    chuan = ''.join(map(lambda x: x.upper(), s.split('-')))

    n=len(chuan)
    rem = n % k
    x=0
    if rem != 0:
        chuan = '?' * (k-rem) + chuan
        x=k-rem
    print(chuan)
    segs = [chuan[i:i+k] for i in range(0, n+x, k)]
    print(chuan,segs, n, rem)
    return '-'.join(segs).replace('?', "")


print(licenseKeyFormatting("5F3Z-2e-9-w", 4))
print(licenseKeyFormatting("2-5g-3-J", 2))
print(licenseKeyFormatting("2-4A0r7-4k", 3))