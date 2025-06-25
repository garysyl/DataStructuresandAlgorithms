# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

2025 spring, Complied by 尚昀霖、数学科学学院



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：

这题做了好几遍都是WA，今天还有别的ddl我先交一会再做

代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：



代码：

```python
import sys
def prim(n, graph):
    INF = float('inf')
    key = [INF] * n
    mst_set = [False] * n
    total = 0
    key[0] = 0

    for _ in range(n):
        min_val = INF
        u = -1
        for i in range(n):
            if not mst_set[i] and key[i] < min_val:
                min_val = key[i]
                u = i

        if u == -1:
            break

        mst_set[u] = True
        total += min_val
        for v in range(n):
            if not mst_set[v] and graph[u][v] < key[v]:
                key[v] = graph[u][v]

    return int(total)
data = list(map(int, sys.stdin.read().split()))
ptr = 0
while ptr < len(data):
    N = data[ptr]
    if ptr + 1 + N * N > len(data):
        break
    ptr += 1
    adj_values = data[ptr:ptr + N * N]
    ptr += N * N
    graph = []
    for i in range(N):
        start = i * N
        end = start + N
        graph.append(adj_values[start:end])
    print(prim(N, graph))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527223045896](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250527223045896.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：



代码：

```python
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        if matrix[-1][-1] == '#':
            return -1

        m, n = len(matrix), len(matrix[0])
        pos = defaultdict(list)
        for i, row in enumerate(matrix):
            for j, c in enumerate(row):
                if c.isupper():
                    pos[c].append((i, j))

        DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0
        q = deque([(0, 0)])

        while q:
            x, y = q.popleft()
            d = dis[x][y]

            if x == m - 1 and y == n - 1: 
                return d

            c = matrix[x][y]
            if c in pos:
                for px, py in pos[c]:
                    if d < dis[px][py]:
                        dis[px][py] = d
                        q.appendleft((px, py))
                del pos[c]  
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and d + 1 < dis[nx][ny]:
                    dis[nx][ny] = d + 1
                    q.append((nx, ny))

        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527225058476](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250527225058476.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：



代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        INF = float('inf')
        prev = [INF] * n
        prev[src] = 0
        
        for _ in range(K + 1):
            curr = list(prev)
            for f, t, price in flights:
                if prev[f] != INF:
                    curr[t] = min(curr[t], prev[f] + price)
            prev = curr
        
        return -1 if prev[dst] == INF else prev[dst]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-05-27 224003.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：



代码：

```python
import sys
from collections import deque
data = list(map(int, sys.stdin.read().split()))
ptr = 0
N = data[ptr]
ptr += 1
M = data[ptr]
ptr += 1

adj = [[] for _ in range(N + 1)]
for _ in range(M):
    A = data[ptr]
    ptr += 1
    B = data[ptr]
    ptr += 1
    c = data[ptr]
    ptr += 1
    adj[A].append((B, c))

INF = float('inf')
dist = [INF] * (N + 1)
dist[1] = 0
in_queue = [False] * (N + 1)
q = deque()
q.append(1)
in_queue[1] = True

while q:
    u = q.popleft()
    in_queue[u] = False
    for v, w in adj[u]:
        if dist[v] > dist[u] + w:
            dist[v] = dist[u] + w
            if not in_queue[v]:
                if q and dist[v] < dist[q[0]]:
                    q.appendleft(v)
                else:
                    q.append(v)
                in_queue[v] = True

print(dist[N])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527230802374](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250527230802374.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：



代码：

```python
from collections import deque

n, m = map(int, input().split())
adj = [[] for _ in range(n)]
in_degree = [0] * n

for _ in range(m):
    a, b = map(int, input().split())
    adj[a].append(b)
    in_degree[b] += 1

queue = deque()
top_order = []

for i in range(n):
    if in_degree[i] == 0:
        queue.append(i)

while queue:
    u = queue.popleft()
    top_order.append(u)
    for v in adj[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
            queue.append(v)

levels = [0] * n
for u in reversed(top_order):
    for v in adj[u]:
        if levels[u] < levels[v] + 1:
            levels[u] = levels[v] + 1

total = sum(100 + level for level in levels)
print(total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527225405336](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250527225405336.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周我通过做作业和每日选做深化了图论算法（MST、最短路径、拓扑排序）与哈希冲突处理的理解，提升了复杂问题建模与代码调试能力，尤其体会到：灵活选择算法、精准处理边界条件、抽象问题模型是高效解题的关键。





