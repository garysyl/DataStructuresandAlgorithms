# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

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

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：

![image-20250401154509427](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401154509427.png)

代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path):
            if len(path) == len(nums):
                result.append(path)
                return
            for i in range(len(nums)):
                if nums[i] not in path:
                    backtrack(path + [nums[i]])

        result = []
        backtrack([])
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：



代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0]:
            return False
        m, n = len(board), len(board[0])
        def dfs(x, y, index):
            if index == len(word):
                return True
            if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[index]:
                return False
            temp, board[x][y] = board[x][y], '#'
            found = (dfs(x + 1, y, index + 1) or
                     dfs(x - 1, y, index + 1) or
                     dfs(x, y + 1, index + 1) or
                     dfs(x, y - 1, index + 1))
            board[x][y] = temp
            return found
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401154235688](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401154235688.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：



代码：

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        def traverse(node):
            if node is None:
                return
            traverse(node.left)
            result.append(node.val)
            traverse(node.right)
        traverse(root)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401153523044](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401153523044.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：



代码：

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(current_level)

        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401152746465](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401152746465.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：



代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def is_palindrome(sub):
            return sub == sub[::-1]

        def backtrack(start, path):
            if start == len(s):
                result.append(path[:])
                return
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if is_palindrome(substring):
                    path.append(substring)
                    backtrack(end, path)
                    path.pop()

        result = []
        backtrack(0, [])
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401152548783](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401152548783.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：

用一个叫**OrderedDict**的东西可以在 O(1) 时间复杂度内进行移动元素的操作，这样可以在更短的时间内通过，代码也更简洁

代码：

```python
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401150902321](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250401150902321.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

作为数竞生学树的时候有种似曾相识的感觉，但代码完成的不够熟练，意识到自己要多做题，通过做题加深了对bfs, dfs的理解。









