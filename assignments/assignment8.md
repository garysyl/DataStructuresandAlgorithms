# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：



代码：

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None

        mid = len(nums) // 2  
        root = TreeNode(nums[mid]) 

        root.left = self.sortedArrayToBST(nums[:mid]) 
        root.right = self.sortedArrayToBST(nums[mid+1:]) 
        return root
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235123185](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250416235123185.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：



代码：

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

n = int(input())
nodes = {}

for _ in range(n):
    parts = list(map(int, input().split()))
    val = parts[0]
    child_vals = parts[1:]
    if val not in nodes:
        nodes[val] = TreeNode(val)
    node = nodes[val]
    for child_val in child_vals:
        if child_val not in nodes:
            nodes[child_val] = TreeNode(child_val)
        child_node = nodes[child_val]
        node.children.append(child_node)

child_values = set()
for node in nodes.values():
    for child in node.children:
        child_values.add(child.value)

root = None
for node in nodes.values():
    if node.value not in child_values:
        root = node
        break

def traverse(node):
    sorted_children = sorted(node.children, key=lambda x: x.value)
    smaller = [child for child in sorted_children if child.value < node.value]
    larger = [child for child in sorted_children if child.value > node.value]
    for child in smaller:
        traverse(child)
    print(node.value)
    for child in larger:
        traverse(child)

traverse(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235201425](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250416235201425.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：



代码：

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:

        def dfs(node, current_sum):
            if not node:
                return 0

            current_sum = current_sum * 10 + node.val

            if not node.left and not node.right:
                return current_sum

            return dfs(node.left, current_sum) + dfs(node.right, current_sum)

        return dfs(root, 0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235330333](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250416235330333.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：



代码：

```python
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None

    root_val = preorder[0]
    root_index_inorder = inorder.index(root_val)

    root = TreeNode(root_val)

    root.left = build_tree(preorder[1:root_index_inorder + 1], inorder[:root_index_inorder])

    root.right = build_tree(preorder[root_index_inorder + 1:], inorder[root_index_inorder + 1:])

    return root

def postorder_traversal(root):
    if not root:
        return ""

    left_subtree = postorder_traversal(root.left)
    right_subtree = postorder_traversal(root.right)

    return left_subtree + right_subtree + root.val

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

while True:
    try:
        preorder = input()
        inorder = input()

        root = build_tree(preorder, inorder)
        postorder = postorder_traversal(root)

        print(postorder)
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235354049](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250416235354049.png)



### T24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：



代码：

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []

def split_by_comma(s):
    result = []
    current = []
    balance = 0
    for c in s:
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
        elif c == ',' and balance == 0:
            result.append(''.join(current))
            current = []
            continue
        current.append(c)
    if current:
        result.append(''.join(current))
    return result

def parse(s):
    if not s:
        return None
    root = TreeNode(s[0])
    if len(s) > 1 and s[1] == '(':
        sub_str = s[2:-1]
        children_strs = split_by_comma(sub_str)
        for child_str in children_strs:
            root.children.append(parse(child_str))
    return root

def preorder_traversal(root):
    result = []
    def helper(node):
        if not node:
            return
        result.append(node.val)
        for child in node.children:
            helper(child)
    helper(root)
    return ''.join(result)

def postorder_traversal(root):
    result = []
    def helper(node):
        if not node:
            return
        for child in node.children:
            helper(child)
        result.append(node.val)
    helper(root)
    return ''.join(result)

s = input().strip()
root = parse(s)
print(preorder_traversal(root))
print(postorder_traversal(root))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235502397](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250416235502397.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：

真的做不完了我先交然后再做一会儿

代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

建树还不熟，课上听那个完全二叉树的题目感觉梦回数竞，但数算毕竟有数据结构和算法两块，我由于是数院的算法还行，但是数据结构确实太差了还得多练。









