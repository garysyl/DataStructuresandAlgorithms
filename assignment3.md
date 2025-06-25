# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by 尚昀霖 数学科学学院



> **说明：**
>
> 1. **惊蛰⽉考**：AC3 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：

一开始没考虑.@浪费很多时间，20min

代码：

```python
def ismail(name):
    if not name:
        return 'NO'
    splitname = name.split('@')
    if len(splitname) != 2:
        return 'NO'
    username, domain = splitname
    if not username:
        return 'NO'
    sdomain=domain.split('.')
    if len(sdomain)==1:
        return 'NO'
    if sdomain[0]=='':
        return 'NO'
    if name[0] in '@.' or name[-1] in '@.':
        return 'NO'
    if username[-1]=='.':
        return 'NO'
    return 'YES'

import sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    print(ismail(line))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250311155331277](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311155331277.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：

排序找老二，15min

代码：

```python
import sys
input = sys.stdin.read().split()
idx = 0
while True:
    N = int(input[idx])
    M = int(input[idx+1])
    idx +=2
    if N ==0 and M ==0:
        break
    count = {}
    for _ in range(N):
        rank = input[idx:idx+M]
        idx +=M
        for p in rank:
            count[p] = count.get(p,0) +1
    max_count = max(count.values())
    second_max = max(v for v in count.values() if v < max_count)
    second_players = sorted(int(k) for k,v in count.items() if v == second_max)
    print(' '.join(map(str,second_players)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![屏幕截图 2025-03-11 232628](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-03-11 232628.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：

枚举可能投放点，计算各点垃圾量，记录最大值及出现次数。25min

代码：

```python
d = int(input())
n = int(input())
garbage = []
for _ in range(n):
    x, y, i = map(int, input().split())
    garbage.append((x, y, i))

if n == 0:
    print("0 0")
else:
    min_x = min(g[0] for g in garbage)
    max_x = max(g[0] for g in garbage)
    min_y = min(g[1] for g in garbage)
    max_y = max(g[1] for g in garbage)

    start_x = max(0, min_x - d)
    end_x = min(1024, max_x + d)
    start_y = max(0, min_y - d)
    end_y = min(1024, max_y + d)

    max_total = -1
    count = 0

    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            current_total = 0
            for (gx, gy, gi) in garbage:
                if (gx >= x - d) and (gx <= x + d) and (gy >= y - d) and (gy <= y + d):
                    current_total += gi
            if current_total > max_total:
                max_total = current_total
                count = 1
            elif current_total == max_total:
                count += 1

    print(f"{count} {max_total}")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250311233431239](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311233431239.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：

困难的题目！场下完成时严重超时，我用优先队列（最小堆）来逐步生成可能的和，并每次只保留最小的n个。每次合并两个序列时，利用堆高效获取最小值，并记录已处理的组合避免重复。这样逐步合并所有序列，最终得到最小的n个和。

代码：

```python
import heapq

def merge(a, b, n):
    heap = []
    visited = set()
    heapq.heappush(heap, (a[0] + b[0], 0, 0))
    visited.add((0, 0))
    res = []
    while len(res) < n and heap:
        val, i, j = heapq.heappop(heap)
        res.append(val)
        if i + 1 < len(a) and (i + 1, j) not in visited:
            heapq.heappush(heap, (a[i + 1] + b[j], i + 1, j))
            visited.add((i + 1, j))
        if j + 1 < len(b) and (i, j + 1) not in visited:
            heapq.heappush(heap, (a[i] + b[j + 1], i, j + 1))
            visited.add((i, j + 1))
    return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        m = int(input[ptr])
        n = int(input[ptr + 1])
        ptr += 2
        sequences = []
        for _ in range(m):
            seq = list(map(int, input[ptr:ptr + n]))
            ptr += n
            seq.sort()
            sequences.append(seq)
        if m == 0:
            print()
            continue
        current = sequences[0]
        for i in range(1, m):
            current = merge(current, sequences[i], n)
        print(' '.join(map(str, current)))

if __name__ == "__main__":
    main()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250311235115376](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250311235115376.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

时限内AC3，很失望，反反复复那道题居然因为奇未知原因WA，并且作为国际象棋爱好者居然没有做出那个回溯算法，后来花时间弄明白了，决定要多下功夫了。开始追赶每日练习。







