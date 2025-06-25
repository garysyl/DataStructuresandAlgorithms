# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：

通过递归比较两个链表的当前节点值，将较小的节点连接到结果链表中，并继续递归处理剩余节点，直到其中一个链表为空，然后将非空链表直接连接到结果链表的末尾

代码：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        elif not list2:
            return list1
        elif list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325224906892](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325224906892.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>



代码：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        slow = prev
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node

        return True
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325225457897](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325225457897.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class ListNode:
    def __init__(self, url: str):
        self.url = url
        self.next = None
        self.prev = None

class BrowserHistory:

    def __init__(self, homepage: str):
        self.current = ListNode(homepage)

    def visit(self, url: str) -> None:
        new_node = ListNode(url)
        self.current.next = new_node
        new_node.prev = self.current
        self.current = new_node

    def back(self, steps: int) -> str:
        while self.current.prev and steps > 0:
            self.current = self.current.prev
            steps -= 1
        return self.current.url

    def forward(self, steps: int) -> str:
        while self.current.next and steps > 0:
            self.current = self.current.next
            steps -= 1
        return self.current.url
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325225812464](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325225812464.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：

遍历输入的中序表达式，逐个字符处理。遇到数字时，完整读取整个数字（包括小数点）并加入输出列表。遇到左括号时，直接压入栈中。遇到右括号时，弹出栈中的运算符直到遇到左括号，并将这些运算符加入输出列表。遇到运算符时，根据优先级规则弹出栈中优先级更高或相等的运算符，并将当前运算符压入栈中。最后，将栈中剩余的运算符全部弹出并加入输出列表。

代码：

```python
n = int(input())
for _ in range(n):
    s = input().strip()
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    stack = []
    output = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isdigit() or c == '.':
            num = []
            while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                num.append(s[i])
                i += 1
            output.append(''.join(num))
        elif c == '(':
            stack.append(c)
            i += 1
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack:
                stack.pop()  
            i += 1
        else: 
            while stack and stack[-1] != '(' and precedence[c] <= precedence.get(stack[-1], 0):
                output.append(stack.pop())
            stack.append(c)
            i += 1
    while stack:
        output.append(stack.pop())
    print(' '.join(output))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325231152898](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325231152898.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
from collections import deque

def josephus_problem(n, p, m):
    if n == 0 or p == 0 or m == 0:
        return ""
    q = deque(range(1, n + 1))
    for _ in range(p - 1):
        q.append(q.popleft())

    result = []
    while q:
        for _ in range(m - 1):
            q.append(q.popleft())
        result.append(str(q.popleft()))

    return ",".join(result)
while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    print(josephus_problem(n, p, m))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325231528005](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325231528005.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：



代码：

```python
def count_overtakes(n, speeds):
    def merge_sort(arr, temp, left, right):
        nonlocal count
        if left >= right:
            return
        mid = (left + right) // 2
        merge_sort(arr, temp, left, mid)
        merge_sort(arr, temp, mid + 1, right)
        i, j, k = left, mid + 1, left
        while i <= mid and j <= right:
            if arr[i] < arr[j]:
                temp[k] = arr[i]
                count += (right - j + 1)
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
            k += 1
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        for idx in range(left, right + 1):
            arr[idx] = temp[idx]
    
    count = 0
    temp = [0] * n
    merge_sort(speeds, temp, 0, n - 1)
    return count

n = int(input())
speeds = [int(input()) for _ in range(n)]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325233653853](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250325233653853.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

在做每日选做之余我去接着尝试了下本地部署大模型。我发现之前用lmstudio部署时都是在cpu上跑，效率很低，由于我的设备是英特尔核显，想用gpu的话得用ollama，我又去学了下，并且为了使能在核显上正常运行还需要下ipex-llm，费了很大劲之后终于成功了，用它尝试了下最新的大模型gemma3，感觉比起废话很多的deepseek要好用些。我测试用的是gemma3:27b-q3量化（因为q4太大了内存装不下），能AC大部分力扣的题目，而且速度还算快，好像还能识别图片，我接下来打算学学怎么实现。

以下是gemma3:27b-q3完成力扣10正则表达式匹配（困难）的对话框

![屏幕截图 2025-03-25 222823](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-03-25 222823.png)

ac截图：

![屏幕截图 2025-03-25 224902](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-03-25 224902.png)

代码：

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)
        
        # 创建一个二维 DP 数组，dp[i][j] 表示 s 前 i 个字符是否能被 p 前 j 个字符匹配
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # 初始化：空字符串和空正则表达式匹配
        dp[0][0] = True
        
        # 处理正则表达式开头为 '*' 的情况
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        # 填充 DP 表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 当前字符匹配，可以是 '.' 或者字符相同
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    # '*' 匹配零个或多个前面的元素，需要考虑两种情况
                    # 1. 匹配零个：则当前状态由 j-2 决定（因为 '*' 占据两个位置）
                    # 2. 匹配一个或多个：则看前一状态是否匹配，并且当前字符与 '*' 前的字符匹配
                    dp[i][j] = dp[i][j-2]
                    if p[j-2] == '.' or s[i-1] == p[j-2]:
                        dp[i][j] |= dp[i-1][j]
                else:
                    # 当前字符不匹配且不是特殊符号，则无法匹配
                    dp[i][j] = False
        
        return dp[m][n]
```







