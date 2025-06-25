# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

2025 spring, Complied by 数学科学学院，尚昀霖



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

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        depth = 0
        node = root
        while node.left:
            depth += 1
            node = node.left
        
        if depth == 0:
            return 1
        
        left = 0
        right = (1 << depth) - 1  
        
        def exists(index):
            path = format(index, f'0{depth}b')
            current = root
            for bit in path:
                if bit == '0':
                    current = current.left
                else:
                    current = current.right
                if not current:
                    return False
            return True
        
        while left <= right:
            mid = (left + right) // 2
            if exists(mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return (1 << depth) - 1 + (right + 1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422233550163](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422233550163.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：



代码：

```python
from collections import deque
from typing import Optional

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        level = 0
        
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
            
            if level % 2 == 0:
                result.append(current_level)
            else:
                result.append(current_level[::-1])
            
            level += 1
        
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422233755600](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422233755600.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：



代码：

```python
import heapq

def calculate_min_path_length():
    n = int(input())
    weights = list(map(int, input().split()))
    
    heapq.heapify(weights)
    
    total = 0
    while len(weights) > 1:
        a = heapq.heappop(weights)
        b = heapq.heappop(weights)
        merged = a + b
        total += merged
        heapq.heappush(weights, merged)
    
    print(total)

calculate_min_path_length()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422234016175](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422234016175.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：



代码：

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value):
    if root is None:
        return Node(value)
    current = root
    parent = None
    while current is not None:
        parent = current
        if value < current.value:
            current = current.left
        elif value > current.value:
            current = current.right
        else:
            return root  
    if value < parent.value:
        parent.left = Node(value)
    else:
        parent.right = Node(value)
    return root

def level_order(root):
    from collections import deque
    result = []
    if not root:
        return result
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(str(node.value))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

nums = list(map(int, input().split()))
root = None
for num in nums:
    root = insert(root, num)

output = level_order(root)
print(' '.join(output))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422234608389](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422234608389.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：



代码：

```python
import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    heap = []
    output = []
    index = 1  
    
    for _ in range(n):
        op_type = int(data[index])
        index += 1
        if op_type == 1:
            value = int(data[index])
            index += 1
            heapq.heappush(heap, value)
        else:
            min_val = heapq.heappop(heap)
            output.append(str(min_val))
    
    print('\n'.join(output))

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235501151](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422235501151.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：



代码：

```python
import heapq

class Node:
    def __init__(self, value, char=None):
        self.value = value
        self.char = char
        self.left = None
        self.right = None
        self.min_char = char 
    def __lt__(self, other):
        if self.value == other.value:
            return self.min_char < other.min_char
        return self.value < other.value

def build_huffman_tree(freq_dict):
    heap = []
    for char, freq in freq_dict.items():
        node = Node(freq, char)
        heapq.heappush(heap, node)
    
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = Node(a.value + b.value)
        if a < b:
            merged.left, merged.right = a, b
        else:
            merged.left, merged.right = b, a
        merged.min_char = min(a.min_char, b.min_char)
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def build_encoding_table(root):
    encoding_table = {}
    def traverse(node, code):
        if node is None:
            return
        if node.left is None and node.right is None:
            encoding_table[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
    traverse(root, '')
    return encoding_table

def is_binary(s):
    return all(c in '01' for c in s)

def main():
    import sys
    input = sys.stdin.read().splitlines()
    n = int(input[0])
    freq_dict = {}
    for i in range(1, n+1):
        parts = input[i].split()
        char = parts[0]
        freq = int(parts[1])
        freq_dict[char] = freq
    process_lines = input[n+1:]
    
    root = build_huffman_tree(freq_dict)
    if root:
        encoding_table = build_encoding_table(root)
    else:
        encoding_table = {}
    
    for line in process_lines:
        line = line.strip()
        if is_binary(line):
            decoded = []
            node = root
            for bit in line:
                if node is None:
                    break
                if bit == '0':
                    node = node.left
                else:
                    node = node.right
                if node.left is None and node.right is None:
                    decoded.append(node.char)
                    node = root
            print(''.join(decoded))
        else:
            encoded = []
            error = False
            for c in line:
                if c not in encoding_table:
                    error = True
                    break
                encoded.append(encoding_table[c])
            if error:
                print('') 
            else:
                print(''.join(encoded))

if __name__ == '__main__':
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235339439](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250422235339439.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

期中刚结束，正在补课程回放和追赶每日选做。目前正在摸索把自己的数学思维转化成代码的方式，但最后发现还是必须靠练题积累经验。









