# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

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

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：



代码：

```python
from collections import deque

T = int(input())
for _ in range(T):
    R, C = map(int, input().split())
    maze = []
    start = end = None
    for i in range(R):
        row = input().strip()
        maze.append(row)
        if 'S' in row:
            start = (i, row.index('S'))
        if 'E' in row:
            end = (i, row.index('E'))
    
    visited = [[False]*C for _ in range(R)]
    q = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True
    found = False
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    
    while q:
        x, y, steps = q.popleft()
        if (x, y) == end:
            print(steps)
            found = True
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < R and 0 <= ny < C and not visited[nx][ny] and maze[nx][ny] != '#':
                visited[nx][ny] = True
                q.append((nx, ny, steps + 1))
    
    if not found:
        print("oop!")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506220946850](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506220946850.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：



代码：

```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        if n == 0:
            return []
        group = [0] * n
        for i in range(1, n):
            if nums[i] - nums[i-1] > maxDiff:
                group[i] = i
            else:
                group[i] = group[i-1]
        answer = []
        for u, v in queries:
            if u == v:
                answer.append(True)
            else:
                answer.append(group[u] == group[v])
        return answer
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506221738510](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506221738510.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：



代码：

```python
b=list(map(float,input().split()))
left=0
right=1000000000
n=len(b)
b.sort()
m = int(n * 0.4)
x = b[m]
while True:
    mid = (left + right) // 2
    if mid==left:
        break
    if mid*x/1000000000+1.1**(mid*x/1000000000)<85:
        left=mid
    else:
        right=mid
print(right)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506164629697](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506164629697.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：



代码：

```python
def has_cycle(n, edges):
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    visited = [False] * n
    recursion_stack = [False] * n

    def dfs(node):
        visited[node] = True
        recursion_stack[node] = True

        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor):
                    return True
            elif recursion_stack[neighbor]:
                return True

        recursion_stack[node] = False
        return False

    for i in range(n):
        if not visited[i]:
            if dfs(i):
                return "Yes"

    return "No"
n,m=map(int,input().split())
edges=[]
for _ in range(m):
    edges.append(map(int,input().split()))
print(has_cycle(n,edges))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506204553014](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506204553014.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：



代码：

```python
import sys
P = int(sys.stdin.readline())
locations = [sys.stdin.readline().strip() for _ in range(P)]
loc_to_idx = {loc: idx for idx, loc in enumerate(locations)}

INF = float('inf')
dist = [[INF] * P for _ in range(P)]
next_node = [[-1] * P for _ in range(P)]

for i in range(P):
    dist[i][i] = 0
    next_node[i][i] = i

Q = int(sys.stdin.readline())
for _ in range(Q):
    parts = sys.stdin.readline().strip().split()
    a, b, d = parts[0], parts[1], int(parts[2])
    i, j = loc_to_idx[a], loc_to_idx[b]
    if d < dist[i][j]:
        dist[i][j] = d
        dist[j][i] = d
        next_node[i][j] = j
        next_node[j][i] = i

for k in range(P):
    for i in range(P):
        for j in range(P):
            if dist[i][k] + dist[k][j] < dist[i][j]:
                dist[i][j] = dist[i][k] + dist[k][j]
                next_node[i][j] = next_node[i][k]

R = int(sys.stdin.readline())
for _ in range(R):
    start, end = sys.stdin.readline().strip().split()
    if start == end:
        print(start)
        continue

    i, j = loc_to_idx[start], loc_to_idx[end]
    if next_node[i][j] == -1:
        print("No path exists")
        continue

    path = []
    current = i
    while current != j:
        path.append(locations[current])
        next_current = next_node[current][j]
        path.append(str(dist[current][next_current]))
        current = next_current
    path.append(locations[j])

    formatted = []
    for k in range(0, len(path) - 1, 2):
        formatted.append(f"{path[k]}->({path[k + 1]})")
    formatted.append(path[-1])
    print("->".join(formatted))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506214910170](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506214910170.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：



代码：

```python
def is_valid_move(x, y, n, board):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def get_next_moves_count(x, y, n, board, move_x, move_y):
    count = 0
    for k in range(8):
        next_x = x + move_x[k]
        next_y = y + move_y[k]
        if is_valid_move(next_x, next_y, n, board):
            count += 1
    return count

def solve_knights_tour(n, sr, sc):
    move_x = [2, 2, -2, -2, 1, 1, -1, -1]
    move_y = [1, -1, 1, -1, 2, -2, 2, -2]

    board = [[-1 for _ in range(n)] for _ in range(n)]

    board[sr][sc] = 0

    def solve(x, y, movei):
        if movei == n * n:
            return True

        next_moves = []

        for k in range(8):
            next_x = x + move_x[k]
            next_y = y + move_y[k]

            if is_valid_move(next_x, next_y, n, board):
                count = get_next_moves_count(next_x, next_y, n, board, move_x, move_y)
                next_moves.append((count, next_x, next_y))

        next_moves.sort()

        for _, next_x, next_y in next_moves:
            board[next_x][next_y] = movei
            if solve(next_x, next_y, movei + 1):
                return True
            board[next_x][next_y] = -1

        return False

    if solve(sr, sc, 1):
        return "success"
    else:
        return "fail"
n = int(input().strip())
sr, sc = map(int, input().strip().split())

result = solve_knights_tour(n, sr, sc)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506215556388](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250506215556388.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

学了Warnsdorff，复习了dijkstra，通过做leetcode每日一题复习了点数学里的递推计数，目前做二分查找已经可以做到很快了，但是对自己bfs和dfs的速度还不算满意。









