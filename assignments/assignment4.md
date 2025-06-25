# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution:
    def singleNumber(self, nums):
        result = 0
        for num in nums:
            result ^= num
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![屏幕截图 2025-03-17 205018](C:\Users\garys\Pictures\Screenshots\屏幕截图 2025-03-17 205018.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：

用两个栈，一个存数字，一个存字符串。遇到数字就存起来，遇到左括号就把数字和当前字符串压栈，然后重置。遇到右括号就把当前字符串重复指定次数，再和栈顶的字符串拼接。普通字符直接加到当前字符串里。

代码：

```python
def decompress_string(s):
    num_stack = []
    str_stack = []
    current_num = 0
    current_str = []

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            if current_num == 0:
                current_num = 1
            num_stack.append(current_num)
            str_stack.append(''.join(current_str))
            current_num = 0
            current_str = []
        elif char == ']':
            if current_num == 0:
                current_num = 1
            repeated_str = ''.join(current_str * current_num)
            current_num = num_stack.pop()
            current_str = list(str_stack.pop() + repeated_str)
        else:
            current_str.append(char)

    return ''.join(current_str)


input_str = input()
output_str = decompress_string(input_str)
print(output_str)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318225056583](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250318225056583.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：

用双指针法。两个链表如果相交，那么它们的尾部节点应该是相同的。通过遍历两个链表来找到它们的尾节点，并比较它们的长度，然后重置指针到每个链表的头节点，并通过移动较长链表的指针来使两个链表的长度相同。最后同时遍历两个链表，直到找到相交节点或到达链表末尾。

代码：

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        def get_length(head):
            length = 0
            while head:
                length += 1
                head = head.next
            return length
        lenA = get_length(headA)
        lenB = get_length(headB)
        pA, pB = headA, headB
        if lenA > lenB:
            for _ in range(lenA - lenB):
                pA = pA.next
        else:
            for _ in range(lenB - lenA):
                pB = pB.next
        while pA and pB:
            if pA == pB:
                return pA
            pA = pA.next
            pB = pB.next
        return None
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318225718728](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250318225718728.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：

prev 用来保存前一个节点。current 用于遍历链表的当前节点。在每次迭代中暂存当前节点的下一个节点（temp_next），然后反转当前节点的指针指向它的前一个节点（prev），接着更新 prev 为当前节点，最后将 current 移动到下一个节点。当 current 为 None 时，遍历结束，返回 prev，此时 prev 就是反转后的头节点。时间复杂度O(n)，空间复杂度 O(1)。

代码：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        
        while current is not None:
            temp_next = current.next  
            current.next = prev 
            prev = current  
            current = temp_next  
        return prev

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318230203856](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250318230203856.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：

先对 nums1 和 nums2 的元素按 nums1 的值进行排序，并将相同 nums1 值的元素分到同一组。然后使用一个辅助类 MaxSumK，它维护一个大小为 k 的最小堆，用于动态计算当前最大的 k 个值的和。在遍历每个分组时，先为组内的每个元素设置 answer[i] 为当前的最大和，然后将当前元素的 nums2 值加入 MaxSumK，更新堆和总和。

代码：

```python
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        elements = list(zip(nums1, nums2, range(n)))
        elements.sort(key=lambda x: x[0])
        groups = []
        current_group = []
        current_val = None
        for elem in elements:
            if elem[0] != current_val:
                if current_group:
                    groups.append(current_group)
                current_val = elem[0]
                current_group = [elem]
            else:
                current_group.append(elem)
        if current_group:
            groups.append(current_group)
        max_sum_k = MaxSumK(k)
        for group in groups:
            for elem in group:
                original_index = elem[2]
                answer[original_index] = max_sum_k.get_sum()
            for elem in group:
                max_sum_k.add(elem[1])
        return answer

class MaxSumK:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.sum_k = 0
    
    def add(self, num):
        if self.k == 0:
            return
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, num)
            self.sum_k += num
        else:
            if num > self.heap[0]:
                popped = heapq.heappop(self.heap)
                self.sum_k -= popped
                heapq.heappush(self.heap, num)
                self.sum_k += num
    
    def get_sum(self):
        return self.sum_k
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318224837219](C:\Users\garys\AppData\Roaming\Typora\typora-user-images\image-20250318224837219.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这周我把每日一题先搁下了去试了试我更感兴趣的鸢尾花。这次的Q6我还没完全完成怕时间不够先提交上去一会儿接着弄。在尝试神经网络鸢尾花分类时，我对PyTorch框架和神经网络在分类问题中的应用有了一些初步的了解。感觉很好玩。

这次的Q6我大概学会了如何通过调整隐藏层和神经元数量来改变模型的复杂度，以及如何选择合适的学习率和激活函数来提高模型的性能。但是还需要花时间理解，正在探索中...







