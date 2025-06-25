# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by 尚昀霖、数学科学学院



> **说明：**
>
> 1. **⽉考**：AC3 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：



代码：

```python
N, K = map(int, input().split())
cows = []
for i in range(N):
    Ai, Bi = map(int, input().split())
    cows.append((i + 1, Ai, Bi))
cows.sort(key=lambda x: x[1], reverse=True)
top_k_cows = cows[:K]
top_k_cows.sort(key=lambda x: x[2], reverse=True)
print(top_k_cows[0][0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515134557250](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250515134557250.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：

考场上看到这个题想到了数竞学过的Catalan数，所以就直接套公式了，场下好好写了一遍回溯的代码。

代码：

```python
import math

n = int(input())
c = math.factorial(2 * n) // (math.factorial(n) ** 2)
print(c // (n + 1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515134426647](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250515134426647.png)

回溯：

```python
def count_seq(n):
    def backtrack(in_stack, out_stack, total_pushed):
        if in_stack == 0 and out_stack == n:
            return 1
        res = 0
        if total_pushed < n:
            res += backtrack(in_stack + 1, out_stack, total_pushed + 1) # push
        if in_stack > 0:
            res += backtrack(in_stack - 1, out_stack + 1, total_pushed) # pop
        return res

    return backtrack(0, 0, 0)

n = int(input())
print(count_seq(n))
```

![image-20250515134511250](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250515134511250.png)

### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：



代码：

```python
n = int(input())
cards = input().split()

queues_first = [[] for _ in range(10)]  
for card in cards:
    y = int(card[1])
    queues_first[y].append(card)

first_order = []
for i in range(1, 10):
    first_order.extend(queues_first[i])

for i in range(1, 10):
    print(f"Queue{i}:{' '.join(queues_first[i])}")

queues_second = {'A': [], 'B': [], 'C': [], 'D': []}
for card in first_order:
    suit = card[0]
    queues_second[suit].append(card)

for suit in ['A', 'B', 'C', 'D']:
    print(f"Queue{suit}:{' '.join(queues_second[suit])}")

sorted_result = []
for suit in ['A', 'B', 'C', 'D']:
    sorted_result.extend(queues_second[suit])
print(' '.join(sorted_result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250520235508823](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250520235508823.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：



代码：

```python
import heapq

v, a = map(int, input().split())
edges = []
for _ in range(a):
    u, w = map(int, input().split())
    edges.append((u, w))

adj = [[] for _ in range(v + 1)]
in_degree = [0] * (v + 1)

for u, w in edges:
    adj[u].append(w)
    in_degree[w] += 1

queue = []
for i in range(1, v + 1):
    if in_degree[i] == 0:
        heapq.heappush(queue, i)

result = []
while queue:
    u = heapq.heappop(queue)
    result.append(u)
    for w in adj[u]:
        in_degree[w] -= 1
        if in_degree[w] == 0:
            heapq.heappush(queue, w)

result = [f"v{x}" for x in result]
print(" ".join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515134700031](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250515134700031.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：

状态扩展。将原本的“城市”状态扩展为“城市 + 当前已花费的通行费”，形成一个新的状态空间。

代码：

```python
import heapq

K = int(input())
N = int(input())
R = int(input())

adj = [[] for _ in range(N + 1)]
for _ in range(R):
    S, D, L, T = map(int, input().split())
    adj[S].append((D, L, T))

INF = 10**18
dist = [[INF] * (K + 1) for _ in range(N + 1)]
dist[1][0] = 0

heap = []
heapq.heappush(heap, (0, 1, 0))

while heap:
    d, u, c = heapq.heappop(heap)
    if d > dist[u][c]:
        continue
    for v, l, t in adj[u]:
        new_c = c + t
        if new_c > K:
            continue
        new_d = d + l
        if new_d < dist[v][new_c]:
            dist[v][new_c] = new_d
            heapq.heappush(heap, (new_d, v, new_c))

res = min(dist[N][c] for c in range(K + 1))
print(-1 if res == INF else res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250520235153722](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250520235153722.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：



代码：

```python
def dfs(i):
    if i > n:
        return (0, 0)
    left = 2 * i
    right = 2 * i + 1
    l_selected, l_not = dfs(left)
    r_selected, r_not = dfs(right)
    selected = values[i - 1] + l_not + r_not
    not_selected = max(l_selected, l_not) + max(r_selected, r_not)
    return (selected, not_selected)

n = int(input())
values = list(map(int, input().split()))
result = max(dfs(1))
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515175008092](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250515175008092.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

月考发挥失常只AC3，发现之前学的还有很多漏洞没有补，在通过补之前的每日选做来补漏，另外也在顺手做leetcode的每日一题补计概知识点。









