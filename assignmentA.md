# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

2025 spring, Complied by 尚昀霖 数学科学学院



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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：



代码：

```python
class Vertex:
    def __init__(self, id):
        self.id = id
        self.neighbors = set()
    
    def add_neighbor(self, neighbor_id):
        self.neighbors.add(neighbor_id)

class Graph:
    def __init__(self, n):
        self.n = n
        self.vertices = [Vertex(i) for i in range(n)]
    
    def add_edge(self, a, b):
        self.vertices[a].add_neighbor(b)
        self.vertices[b].add_neighbor(a)
    
    def get_adjacency_matrix(self):
        adj = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for neighbor in self.vertices[i].neighbors:
                adj[i][neighbor] = 1
        return adj
    
    def get_degree_matrix(self):
        deg = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            deg[i][i] = len(self.vertices[i].neighbors)
        return deg
    
    def get_laplacian_matrix(self):
        adj = self.get_adjacency_matrix()
        deg = self.get_degree_matrix()
        laplacian = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i == j:
                    row.append(deg[i][j])
                else:
                    row.append(-1 if adj[i][j] == 1 else 0)
            laplacian.append(row)
        return laplacian

n, m = map(int, input().split())
g = Graph(n)
for _ in range(m):
    a, b = map(int, input().split())
    g.add_edge(a, b)
laplacian = g.get_laplacian_matrix()
for row in laplacian:
    print(' '.join(map(str, row)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429234500356](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429234500356.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：



代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        path = []
        self.backtrack(nums, 0, path, result)
        return result
    
    def backtrack(self, nums: List[int], start: int, path: List[int], result: List[List[int]]) -> None:
        result.append(path.copy())
        for i in range(start, len(nums)):
            path.append(nums[i])
            self.backtrack(nums, i + 1, path, result)
            path.pop()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429234939035](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429234939035.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：



代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        letters = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index, path):
            if index == len(digits):
                result.append(path)
                return
            current_digit = digits[index]
            for letter in letters[current_digit]:
                backtrack(index + 1, path + letter)
        backtrack(0, '')
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429235006808](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429235006808.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：



代码：

```python
from collections import Counter
import sys
input = sys.stdin.read().split()
ptr = 0
t = int(input[ptr])
ptr += 1
for _ in range(t):
    n = int(input[ptr])
    ptr += 1
    nums = []
    for __ in range(n):
        nums.append(input[ptr])
        ptr += 1
    count = Counter(nums)
    conflict = False
    for num, c in count.items():
        if c > 1:
            conflict = True
            break
    if conflict:
        print("NO")
        continue
    num_set = set(nums)
    conflict = False
    for num in nums:
        for i in range(1, len(num)):
            prefix = num[:i]
            if prefix in num_set:
                conflict = True
                break
        if conflict:
            break
    print("NO" if conflict else "YES")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429235112035](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429235112035.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：



代码：

```python
from collections import defaultdict, deque

def main():
    n = int(input())
    words = set()
    for _ in range(n):
        word = input().strip()
        words.add(word)
    start, end = input().split()
    
    if start not in words or end not in words:
        print("NO")
        return
    if start == end:
        print(start)
        return
    pattern_dict = defaultdict(list)
    for word in words:
        for i in range(4):
            pattern = word[:i] + '*' + word[i+1:]
            pattern_dict[pattern].append(word)
    adj = defaultdict(set)
    for word in words:
        for i in range(4):
            pattern = word[:i] + '*' + word[i+1:]
            for neighbor in pattern_dict[pattern]:
                if neighbor != word:
                    adj[word].add(neighbor)
    for word in adj:
        adj[word] = list(adj[word])
    queue = deque([start])
    visited = {start: None}
    found = False
    
    while queue:
        current = queue.popleft()
        if current == end:
            found = True
            break
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
    
    if found:
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = visited[current]
        path.reverse()
        print(' '.join(path))
    else:
        print("NO")

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429235419108](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429235419108.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：



代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result = []
        
        def backtrack(state):
            if len(state) == n:
                board = []
                for col in state:
                    row_str = '.' * col + 'Q' + '.' * (n - col - 1)
                    board.append(row_str)
                result.append(board)
                return
            for col in range(n):
                if is_valid(state, col):
                    backtrack(state + [col])
        
        def is_valid(state, col):
            current_row = len(state)
            for i in range(current_row):
                if state[i] == col:
                    return False
                if abs(current_row - i) == abs(col - state[i]):
                    return False
            return True
        
        backtrack([])
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429235708314](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250429235708314.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

最近临近假期课业压力有些大，五一由于课表排的好可以多放几天假，终于可以腾出时间做题了，希望进步可以体现在五一之后的月考上。







